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

import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_PATH = "tafasir_quran_faiss_vectorstore"
MODEL_PATH = "sentence-transformers/paraphrase-albert-small-v2"
MAX_MEMORY_USAGE = 0.75  # Maximum memory usage threshold (75%)

# Initialize environment and configurations
warnings.filterwarnings("ignore")

# Thread pool executor for blocking operations
executor = ThreadPoolExecutor(max_workers=2)

async def create_embeddings():
    """Create embeddings with optimized memory settings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {'device': device}
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
            logger.info("Creating sample embedding")
            # Create a sample embedding to get dimensions
            sample_embedding = embeddings.embed_query("test")
            logger.info("Initializing FAISS index")
            # Initialize FAISS index
            index = faiss.IndexFlatL2(len(sample_embedding))
            logger.info("Creating vector store")
            # Create vector store
            self._vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            logger.info("Vector store initialization successful")
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

async def embed_data(message: str, session_id: str) -> Dict[str, Any]:
    """Embed data into vector store with error handling."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Embedding data for session: {session_id}")

        vector_store = await state_manager.get_vector_store()

        # Create new document with enhanced metadata
        document = Document(
            page_content=message,
            metadata={
                "source": "user",
                "session_id": session_id,
                "type": "conversation",
                "timestamp": str(datetime.now()),
                "is_question": True
            }
        )

        vector_store.add_documents(documents=[document])
        Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(VECTOR_STORE_PATH)

        logger.info(f"Successfully embedded data for session: {session_id}")
        return {"status": "success", "document_name": f"document_{session_id}"}

    except Exception as e:
        logger.error(f"Error in embed_data: {str(e)}")
        return {"status": "error", "message": str(e)}

async def retrieve_embedded_data(message: Optional[str], session_id: str) -> Optional[List[dict]]:
    """Retrieve embedded data and manage chat history."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Retrieving data for session: {session_id}")

        if Path(f"{VECTOR_STORE_PATH}/index.faiss").exists():
            embeddings = await state_manager.get_embeddings()
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vector_store = await state_manager.get_vector_store()

        if message:
            all_docs = list(vector_store.docstore._dict.values())

            # Perform similarity search based on the message
            # First filter all_docs by session_id using attribute access
            filtered_docs = [doc for doc in all_docs if doc.metadata.get("session_id") == session_id]
            
            if not filtered_docs:
                logger.warning(f"No documents found for session_id: {session_id}. Skipping similarity search.")
                # Return a default system message to provide context
                default_docs = [{
                    "content": "You are a helpful assistant specializing in Quranic tafsir.",
                    "metadata": {
                        "source": "system",
                        "session_id": session_id,
                        "type": "system",
                        "timestamp": str(datetime.now()),
                        "is_question": False
                    }
                }]
                return default_docs

            # Create a new FAISS index with filtered documents
            embeddings = await state_manager.get_embeddings()
            
            try:
                filtered_store = FAISS.from_documents(
                    filtered_docs,
                    embeddings,
                    docstore=InMemoryDocstore(),
                )
            except ValueError as ve:
                logger.error(f"Failed to create FAISS index with filtered_docs: {ve}")
                return []

            search_kwargs = {
                "k": 10,
                "fetch_k": 20,
                "score_threshold": 0.5
            }
            retriever = filtered_store.as_retriever(
                search_kwargs=search_kwargs,
                search_type="similarity_score_threshold"
            )
            docs = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: retriever.invoke(message)
            )

            if not docs:
                logger.warning(f"No similar documents found for the given message in session_id: {session_id}.")
        else:
            # Get all documents and filter by session_id using attribute access
            all_docs = list(vector_store.docstore._dict.values())
            docs = [
                doc for doc in all_docs 
                if doc.metadata.get("session_id") == session_id  # Updated access
            ]

            if not docs:
                logger.warning(f"No documents found for session_id: {session_id}. Returning default system message.")
                # Return a default system message to provide context
                default_docs = [{
                    "content": "You are a helpful assistant specializing in Quranic tafsir.",
                    "metadata": {
                        "source": "system",
                        "session_id": session_id,
                        "type": "system",
                        "timestamp": str(datetime.now()),
                        "is_question": False
                    }
                }]
                return default_docs

        logger.info(f"Retrieved {len(docs)} documents for session {session_id}")

        # Format retrieved documents with metadata using attribute access
        formatted_docs = []
        for doc in docs:
            if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                logger.error(f"Document missing required attributes: {doc}")
                continue  # Skip malformed documents
            formatted_docs.append({
                "content": doc.page_content,       # Changed from doc["content"]
                "metadata": doc.metadata          # Changed from doc["metadata"]
            })

        return formatted_docs

    except Exception as e:
        logger.error(f"Error in retrieve_embedded_data: {str(e)}")
        return None

def validate_inputs(message: Optional[str], session_id: str) -> None:
    """Validate input parameters."""
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Invalid session_id provided.")
        
    if message is not None and not isinstance(message, str):
        raise ValueError("Message must be a string.")

# Create singleton instance
state_manager = StateManager()

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