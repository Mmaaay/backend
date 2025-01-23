import gc
import logging
import os
import warnings
from datetime import datetime
from logging import config
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
from dotenv import load_dotenv
load_dotenv()
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
HF_TOKEN = os.getenv("HF_TOKEN")
# Constants
VECTOR_STORE_DIR = "tafasir_quran_faiss_vectorstore"
VECTOR_STORE_PATH = Path(VECTOR_STORE_DIR).absolute()  # Use absolute path
MODEL_PATH = "models/embeddings"

# Initialize environment and configurations
warnings.filterwarnings("ignore")

# Thread pool executor for blocking operations
executor = ThreadPoolExecutor(max_workers=2)


async def create_embeddings():
    """Create embeddings with optimized memory settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device , 'token':HF_TOKEN}
    encode_kwargs = {'normalize_embeddings': True}

    # Clear memory before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Creating embeddings on device: {device}")

    # Load embeddings in executor to avoid blocking event loop
    embeddings = await asyncio.get_event_loop().run_in_executor(
        executor,
        lambda: HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    )

    # Model initialization is handled automatically by HuggingFaceEmbeddings
    # Remove the manual model initialization as it's not needed

    return embeddings


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
        """Initialize vector store asynchronously."""
        logger.info("Initializing vector store")
        try:
            embeddings = await self.get_embeddings()
            
            # Try to load existing vector store first
            if (VECTOR_STORE_PATH / "index.faiss").exists():
                logger.info("Loading existing vector store")
                self._vector_store = FAISS.load_local(
                    folder_path=str(VECTOR_STORE_PATH),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing vector store with {len(self._vector_store.docstore._dict)} documents")
            else:
                # Create new vector store if none exists
                logger.info("Creating new vector store")
                sample_embedding = embeddings.embed_query("test")
                index = faiss.IndexFlatL2(len(sample_embedding))
                self._vector_store = FAISS(
                    embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
                )
                # Ensure directory exists before saving
                VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
                self._vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
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


async def embed_data(message: str,  ai_response:str, session_id: str) -> Dict[str, Any]:
    """Embed data into vector store with error handling."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Embedding data for session: {session_id}")
        
        vector_store = await state_manager.get_vector_store()
        timestamp = str(datetime.now())
        
        # Create new document
        new_document = Document(
            page_content=message,
            metadata={
                "source": "user",
                "session_id": session_id,
                "type": "conversation",
                "message": message,
                "timestamp": timestamp,
                "is_question": True,
                "Assistant" : {
                    "response": ai_response,
                    "timestamp": timestamp
                }
               
            }
        )

        # Generate unique ID
        doc_id = f"document_{session_id}_{timestamp}"
        
        try:
            # Add document and save
            vector_store.add_documents([new_document], ids=[doc_id])
            vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
            logger.info(f"Added and saved document {doc_id}")
            logger.debug(f"Total documents in store: {len(vector_store.docstore._dict)}")
            
            return {
                "status": "success",
                "document_name": doc_id
            }
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            return {"status": "error", "message": str(e)}

    except Exception as e:
        logger.error(f"Error in embed_data: {str(e)}")
        return {"status": "error", "message": str(e)}


async def retrieve_embedded_data(message: Optional[str], session_id: str) -> Optional[List[dict]]:
    """Retrieve embedded data and manage chat history."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Retrieving data for session: {session_id}")

        vector_store = await state_manager.get_vector_store()
        
        retrieved_context = vector_store.similarity_search(message, k=5 ,
                                                           filters={"metadata.session_id": session_id})
        docs_as_dicts = []
        for doc in retrieved_context:
            try:
                doc_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'timestamp': doc.metadata.get('timestamp', '1970-01-01')
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
        
        logger.info(f"Retrieved {len(formatted_docs)} documents for session {session_id}")
        return formatted_docs

    except Exception as e:
        logger.error(f"Error in retrieve_embedded_data: {str(e)}")
        return None


def validate_inputs(message: Optional[str], session_id: str) -> None:
    """Validate input parameters."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID must be a non-empty string")
    if message is not None and not isinstance(message, str):
        raise ValueError("Message must be a string")


# Create singleton instance
state_manager = StateManager()
logger.info("Initializing embeddings")

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
# It's assumed that initialize_services is called during the application's startup
        await state_manager.get_embeddings()
        logger.info("Initializing vector store")
        await state_manager.get_vector_store()
        logger.info("Successfully initialized application")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
# It's assumed that initialize_services is called during the application's startup