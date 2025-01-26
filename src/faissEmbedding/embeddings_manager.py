import gc
import logging
import os
from pathlib import Path
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer
import torch
import concurrent.futures
import tiktoken
import faiss
from dotenv import load_dotenv
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
# Constants
VECTOR_STORE_DIR = "tafasir_quran_faiss_vectorstore"
VECTOR_STORE_PATH = Path(VECTOR_STORE_DIR).absolute()  # Use absolute path
MODEL_PATH = "models/embeddings"

# Initialize environment and configurations
warnings.filterwarnings("ignore")

# Thread pool executor for blocking operations
executor = ThreadPoolExecutor(max_workers=1)
shared_executor = executor

@asynccontextmanager
async def managed_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False)

async def create_embeddings():
    """Create embeddings with optimized memory settings."""
    # Force CPU to reduce memory usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device, 'token': HF_TOKEN}
    encode_kwargs = {'normalize_embeddings': True}

    # Clear memory before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Creating embeddings on device: {device}")

    try:
        async with managed_executor() as executor:
            embeddings = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: HuggingFaceEmbeddings(
                        model_name=MODEL_PATH,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs
                    )
                ),
                timeout=30,
            )
        return embeddings
    except asyncio.TimeoutError:
        logger.error("Embedding creation timed out after 30 seconds")
        raise

# Global semaphore to limit concurrent tasks
CONCURRENCY_LIMIT = 2
task_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

class StateManager:
    """Memory-efficient state manager for embeddings and vector store."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[FAISS] = None
        self._chat_history: Dict[str, List[dict]] = {}

    async def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            async with self._lock:
                if self._embeddings is None:
                    await self._initialize_embeddings()
        return self._embeddings

    async def _initialize_embeddings(self):
        """Initialize embeddings asynchronously."""
        logger.info("Initializing new embeddings")
        try:
            self._embeddings = await create_embeddings()
            logger.info("Embeddings initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    async def get_vector_store(self) -> FAISS:
        if self._vector_store is None:
            async with self._lock:
                if self._vector_store is None:
                    await self._initialize_vector_store()
        return self._vector_store

    async def _initialize_vector_store(self):
        logger.info("Initializing vector store")
        try:
            # Acquire embeddings
            embeddings = await self.get_embeddings()

            # ...existing code...
            # Use shared_executor for all blocking operations
            loop = asyncio.get_event_loop()
            if (VECTOR_STORE_PATH / "index.faiss").exists():
                logger.info("Loading existing vector store")
                self._vector_store = await loop.run_in_executor(
                    shared_executor,
                    lambda: FAISS.load_local(
                        folder_path=str(VECTOR_STORE_PATH),
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True
                    )
                )
                logger.info(f"Loaded existing vector store with {len(self._vector_store.docstore._dict)} documents")
            else:
                logger.info("Creating new vector store")
                sample_embedding = await loop.run_in_executor(
                    shared_executor,
                    lambda: embeddings.embed_query("test")
                )
                index = faiss.IndexFlatL2(len(sample_embedding))
                self._vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
                await loop.run_in_executor(
                    shared_executor,
                    lambda: self._vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
                )
                logger.info("Created and saved new vector store")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    async def clear_cache(self):
        """Clear memory cache and release resources."""
        async with self._lock:
            if self._embeddings is not None:
                del self._embeddings
                self._embeddings = None

            if self._vector_store is not None:
                del self._vector_store
                self._vector_store = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Cleared cache and released memory")

    def get_chat_history(self, session_id: str) -> List[dict]:
        return self._chat_history.get(session_id, [])

    def append_chat_history(self, session_id: str, message: dict):
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(message)

  

async def embed_data(message: str, ai_response: str, user_id: str) -> Dict[str, Any]:
    """Embed data into vector store with error handling."""
    async with task_semaphore:
        try:
            validate_inputs(message, user_id)
            logger.info(f"Embedding data for User: {user_id}")

            

            vector_store = await asyncio.wait_for(
                state_manager.get_vector_store(), timeout=30
            )
            timestamp = str(datetime.now())

            # Create new document
            new_document = Document(
                page_content=message,
                metadata={
                    "source": "user",
                    "user_id": user_id,
                    "type": "conversation",
                    "message": message,
                    "timestamp": timestamp,
                    "is_question": True,
                    "Assistant": {
                        "response": ai_response,
                        "timestamp": timestamp
                    }
                }
            )

            # Generate unique ID
            doc_id = f"document_{user_id}_{timestamp}"

            try:
                async with managed_executor() as executor:
                    # Add document and save
                    vector_store.add_documents([new_document], ids=[doc_id])
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            executor,
                            lambda: vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
                        ),
                        timeout=30
                    )
                # Release references
                del executor

                logger.info(f"Added and saved document {doc_id}")
                logger.debug(f"Total documents in store: {len(vector_store.docstore._dict)}")

                return {
                    "status": "success",
                    "document_name": doc_id
                }
            except asyncio.TimeoutError:
                logger.error("Embedding data operation timed out after 30 seconds")
                return {"status": "error", "message": "Operation timed out."}
            except Exception as e:
                logger.error(f"Error adding document: {str(e)}")
                return {"status": "error", "message": str(e)}

        except Exception as e:
            logger.error(f"Error in embed_data: {str(e)}")
            return {"status": "error", "message": str(e)}

async def retrieve_embedded_data(message: Optional[str], user_id: str) -> Optional[List[dict]]:
    """Retrieve embedded data and manage chat history."""
    async with task_semaphore:
        try:
            validate_inputs(message, user_id)
            logger.info(f"Retrieving data for User: {user_id}")

            vector_store = await asyncio.wait_for(
                state_manager.get_vector_store(), timeout=30
            )
            retrieved_context = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: vector_store.similarity_search_with_score(message, k=1, filters={"user_id": user_id})
                    ),
                    timeout=30
                )
            docs_as_dicts = []
            for doc, score in retrieved_context:
                if score > 0.8:  # Only include results with a score > 0.8
                    try:
                        doc_dict = {
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'timestamp': doc.metadata.get('timestamp', '1970-01-01'),
                        }
                        docs_as_dicts.append(doc_dict)
                        logger.debug(f"Transformed document: {doc_dict}")
                    except Exception as e:
                        logger.error(f"Error transforming document: {e}")

            # Sort transformed documents
            sorted_docs = sorted(
                docs_as_dicts,
                key=lambda x: datetime.fromisoformat(x['timestamp']),
                reverse=True
            )
            
            logger.debug(f"Sorted documents: {sorted_docs}")

            # Extract questions and responses
            current_questions = [doc['metadata']['message'] for doc in sorted_docs]
            ai_responses = [doc['metadata']['Assistant']['response'] for doc in sorted_docs]

            # Format documents
            formatted_docs = []
            formatted_docs.append({
                "user_question": current_questions,
                "ai_response": ai_responses,
                "type": "history"
            })

            logger.info(f"Retrieved {len(formatted_docs)} documents for User {user_id}")
            return formatted_docs

        except asyncio.TimeoutError:
            logger.error("Retrieve embedded data operation timed out after 30 seconds")
            return None
        except Exception as e:
            logger.error(f"Error in retrieve_embedded_data: {str(e)}")
            return None

def validate_inputs(message: Optional[str], user_id: str) -> None:
    """Validate input parameters."""
    if not user_id or not isinstance(user_id, str):
        raise ValueError("User ID must be a non-empty string")
    if message is not None and not isinstance(message, str):
        raise ValueError("Message must be a string")

# Create singleton instance
state_manager = StateManager()
logger.info("Embedding manager initialized")
    

# Initialize the embeddings and vector store asynchronously
async def initialize_services():
    try:
        logger.info("Initializing embeddings")
        await state_manager.get_embeddings()
        logger.info("Initializing vector store")
        await state_manager.get_vector_store()
        logger.info("Successfully initialized application")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
async def embed_system_response(current_question: str, ai_response: str, session_id: str):
    """Embeds the system's response with metadata"""
    result = await embed_data(current_question, ai_response, session_id)
    logger.info(f"embed_system_response returned: {result}")
    
    # Clear cache after embedding
    await state_manager.clear_cache()

    return result
     

# Ensure executor is properly shutdown on application exit
import atexit

@atexit.register
def shutdown_executor():
    executor.shutdown(wait=False)
    logger.info("Executor shutdown successfully")
