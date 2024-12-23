import gc
import logging
import os
import warnings
from datetime import datetime
from logging import config
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, AsyncGenerator
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
VECTOR_STORE_DIR = "tafasir_quran_faiss_vectorstore"
VECTOR_STORE_PATH = Path(VECTOR_STORE_DIR)
INDEX_PATH = VECTOR_STORE_PATH / "index.faiss"
MODEL_PATH = "sentence-transformers/all-MiniLM-L12-v2"
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
            self._vector_store = await load_or_create_vector_store(embeddings)
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


async def check_duplicate_message(vector_store: FAISS, message: str, session_id: str) -> bool:
    """Check if message already exists for this session."""
    search_kwargs = {
        "k": 1,
        "filter": lambda metadata: (
            filter_by_session(metadata, session_id) and 
            metadata.get("content") == message
        )
    }
    results = vector_store.similarity_search_with_score(message, **search_kwargs)
    return len(results) > 0

async def embed_data(message: str, session_id: str) -> Dict[str, Any]:
    """Non-streaming version that uses the streaming implementation"""
    async for status in embed_data_stream(message, session_id):
        if status["status"] in ["success", "error"]:
            return {
                "status": status["status"],
                "document_name": f"document_{session_id}",
                "embedding_status": status["status"]
            }
    return {"status": "error", "message": "Unknown error occurred"}

async def embed_system_response(message: str, session_id: str, question: str) -> Dict[str, Any]:
    """Embed AI system response into vector store."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Embedding system response for session: {session_id}")

        vector_store = await state_manager.get_vector_store()
        
        document = Document(
            page_content=message,
            metadata={
                "source": "system",
                "session_id": session_id,
                "type": "conversation",
                "timestamp": str(datetime.now()),
                "is_question": False,
                "is_response": True,
                "question_reference": question,
                "content": message
            }
        )

        vector_store.add_documents(documents=[document])
        vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
        logger.info(f"Successfully saved system response to vector store")
        return {"status": "success", "embedding_status": "success"}

    except Exception as e:
        logger.error(f"Error in embed_system_response: {str(e)}")
        return {"status": "error", "embedding_status": "error"}


def filter_by_session(metadata: Dict[str, Any], session_id: str) -> bool:
    logger.debug(f"Filtering document with session_id: {metadata.get('session_id')}, target session_id: {session_id}")
    return metadata.get("session_id") == session_id


async def retrieve_embedded_data(message: str, session_id: str) -> Optional[List[dict]]:
    """Retrieve embedded data and manage chat history."""
    try:
        validate_inputs(message, session_id)
        logger.info(f"Retrieving data for session: {session_id}")

        # Load vector store if exists
        if (VECTOR_STORE_PATH / "index.faiss").exists():
            embeddings = await state_manager.get_embeddings()
            vector_store = FAISS.load_local(
                folder_path=str(VECTOR_STORE_PATH),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            vector_store = await state_manager.get_vector_store()

        # Get all documents for the session
        all_docs = []
        for doc_id, doc in vector_store.docstore._dict.items():
            if doc.metadata.get("session_id") == session_id:
                doc.metadata["id"] = doc_id
                if doc.metadata.get("is_response"):
                    doc.metadata["type"] = "answer"
                elif doc.metadata.get("is_question"):
                    doc.metadata["type"] = "question"
                all_docs.append(doc)

        # Sort documents by timestamp
        sorted_docs = sorted(
            all_docs,
            key=lambda x: datetime.fromisoformat(x.metadata.get("timestamp", "1970-01-01")),
            reverse=True
        )

        formatted_docs = []
        for doc in sorted_docs:
            formatted_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "type": "history"  # Keep as history for backwards compatibility
            })
        
        logger.debug(f"Retrieved history for session {session_id}: {formatted_docs}")
        return formatted_docs

    except Exception as e:
        logger.error(f"Error in retrieve_embedded_data: {str(e)}")
        return None


def validate_inputs(message: str, session_id: str) -> None:
    """Validate input parameters."""
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID must be a non-empty string")


# Create singleton instance
state_manager = StateManager()

# Initialize the embeddings and vector store asynchronously
async def initialize_services():
    try:
        logger.info("Initializing embeddings")
        await state_manager.get_embeddings()
        logger.info("Initializing vector store")
        # Optionally reset the vector store on startup
        # await reset_vector_store()  # Uncomment to reset on startup
        await state_manager.get_vector_store()
        logger.info("Successfully initialized application")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise
# It's assumed that initialize_services is called during the application's startup

def check_directory_exists():
    """Ensure vector store directory exists."""
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    logger.info(f"Vector store directory checked: {VECTOR_STORE_PATH}")

async def reset_vector_store():
    """Clear the vector store and its files."""
    try:
        # Delete the vector store files
        if VECTOR_STORE_PATH.exists():
            for file in VECTOR_STORE_PATH.glob('*'):
                file.unlink()
            VECTOR_STORE_PATH.rmdir()
            logger.info("Vector store directory cleared")
        
        # Reset the state manager's vector store
        await state_manager.clear_cache()
        return True
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        return False

async def load_or_create_vector_store(embeddings) -> FAISS:
    """Load existing vector store or create new one."""
    try:
        # Create directory if it doesn't exist
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        
        if (VECTOR_STORE_PATH / "index.faiss").exists():
            logger.info("Loading existing vector store")
            vector_store = FAISS.load_local(
                folder_path=str(VECTOR_STORE_PATH),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            # Debug log all documents
            for doc_id, doc in vector_store.docstore._dict.items():
                logger.debug(f"Loaded document: {doc_id}")
                logger.debug(f"Content: {doc.page_content}")
                logger.debug(f"Metadata: {doc.metadata}")
            
            logger.info(f"Loaded {len(vector_store.docstore._dict)} documents from store")
            return vector_store
        else:
            logger.info("Creating new vector store")
            sample_embedding = embeddings.embed_query("test")
            index = faiss.IndexFlatL2(len(sample_embedding))
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            logger.info("Created new empty vector store")
            return vector_store
    except Exception as e:
        logger.error(f"Error in load_or_create_vector_store: {str(e)}")
        raise

async def embed_data_stream(message: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream the embedding process"""
    try:
        validate_inputs(message, session_id)
        yield {"status": "starting", "message": "Beginning embedding process"}

        vector_store = await state_manager.get_vector_store()
        yield {"status": "processing", "message": "Retrieved vector store"}

        document = Document(
            page_content=message,
            metadata={
                "source": "user",
                "session_id": session_id,
                "type": "conversation",
                "timestamp": str(datetime.now()),
                "is_question": True,
                "content": message
            }
        )

        vector_store.add_documents(documents=[document])
        yield {"status": "processing", "message": "Added document to vector store"}

        check_directory_exists()
        vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
        yield {"status": "success", "message": "Successfully saved to vector store"}

    except Exception as e:
        logger.error(f"Error in embed_data_stream: {str(e)}")
        yield {"status": "error", "message": str(e)}