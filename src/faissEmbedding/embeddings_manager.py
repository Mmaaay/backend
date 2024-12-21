import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from threading import Lock
import gc

import faiss
import torch
from langchain.schema import AIMessage, HumanMessage
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_PATH = "tafasir_quran_faiss_vectorstore"
MODEL_PATH = "sentence-transformers/all-MiniLM-L12-v2"
MAX_MEMORY_USAGE = 0.75  # Maximum memory usage threshold (75%)

# Initialize environment and configurations
warnings.filterwarnings("ignore")

class LowMemoryEmbeddings(HuggingFaceEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            # Clear any unused memory before loading
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load model with minimal memory footprint
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {
                'device': device,
                'use_auth_token': None,
                'trust_remote_code': True,
                'quantization_config': None if device == "cuda" else {'load_in_8bit': True}
            }
            
            super().__init__(
                model_name=MODEL_PATH,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def embed_documents(self, texts):
        self._load_model()
        return super().embed_documents(texts)
    
    def embed_query(self, text):
        self._load_model()
        return super().embed_query(text)

class StateManager:
    """Memory-efficient state manager for embeddings and vector store"""
    _instance = None
    _embeddings = None
    _vector_store = None
    _chat_history = {}
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StateManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            with self._lock:
                if not self._initialized:
                    self._embeddings = None
                    self._vector_store = None
                    self._chat_history = {}
                    self._initialized = True

    def _check_memory(self):
        """Check if memory usage is within acceptable limits"""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            if memory_allocated / memory_reserved > MAX_MEMORY_USAGE:
                self.clear_cache()
                return False
        return True

    @property
    def embeddings(self):
        if self._embeddings is None:
            with self._lock:
                if self._embeddings is None:
                    if not self._check_memory():
                        raise MemoryError("Insufficient memory available")
                    
                    logger.info("Initializing new embeddings")
                    try:
                        self._embeddings = LowMemoryEmbeddings()
                        logger.info("Embeddings initialization successful")
                    except Exception as e:
                        logger.error(f"Failed to initialize embeddings: {str(e)}")
                        raise
        return self._embeddings

    @property
    def vector_store(self):
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    if not self._check_memory():
                        raise MemoryError("Insufficient memory available")
                    
                    logger.info("Initializing vector store")
                    try:
                        embeddings = self.embeddings
                        sample_embedding = embeddings.embed_query("test")
                        index = faiss.IndexFlatL2(len(sample_embedding))
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
        return self._vector_store

    def clear_cache(self):
        """Clear memory cache and release resources"""
        with self._lock:
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

    def get_chat_history(self, session_id: str) -> list:
        return self._chat_history.get(session_id, [])

    def append_chat_history(self, session_id: str, message: dict):
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(message)

# Create singleton instance
state_manager = StateManager()

def validate_inputs(message: str, session_id: str) -> None:
    """Validate input parameters."""
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID must be a non-empty string")

def embed_data(message: str, session_id: str) -> Dict[str, Any]:
    """
    Embed data into vector store with error handling.
    Returns dict with status and additional information.
    """
    try:
        validate_inputs(message, session_id)
        logger.info(f"Embedding data for session: {session_id}")

        vector_store = state_manager.vector_store
        
        # Check for existing document
        existing_document = vector_store.docstore.search(session_id)
        if existing_document is not None:
            return {
                "status": "exists",
                "message": f"Document with session_id {session_id} already exists"
            }

        # Create document with unique name
        document_name = f"document_{session_id}"
        document = Document(
            page_content=message,
            metadata={
                "source": "user",
                "id": session_id,
                "name": document_name,
                "timestamp": str(datetime.now())
            }
        )

        # Ensure vector store directory exists
        Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)

        # Add document and save
        vector_store.add_documents(documents=[document], ids=[session_id])
        vector_store.save_local(VECTOR_STORE_PATH)

        logger.info(f"Successfully embedded data for session: {session_id}")
        return {"status": "success", "document_name": document_name}

    except Exception as e:
        logger.error(f"Error in embed_data: {str(e)}")
        return {"status": "error", "message": str(e)}

def retrieve_embedded_data(message: str, session_id: str) -> Optional[list]:
    """
    Retrieve embedded data and manage chat history.
    Returns chat history or None if operation fails.
    """
    try:
        validate_inputs(message, session_id)
        logger.info(f"Retrieving data for session: {session_id}")

        # Load or use existing vector store
        faiss_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings=state_manager.embeddings,
            allow_dangerous_deserialization=True
        ) if Path(f"{VECTOR_STORE_PATH}/index.faiss").exists() else state_manager.vector_store

        retriever = faiss_store.as_retriever()

        # Add user message to chat history
        state_manager.append_chat_history(session_id, {
            "role": "human",
            "content": message,
            "timestamp": str(datetime.now())
        })

        # Retrieve relevant documents
        search_results = retriever.get_relevant_documents(query=message)
        retrieved_texts = [doc.metadata.get('answer', 'No answer found') 
                         for doc in search_results]

        # Process messages
        input_messages = [HumanMessage(content=message)]
        dynamic_config = {**config, "configurable": {"thread_id": session_id}}

        # Get model response
        output = app.invoke(
            {
                "messages": input_messages,
                "retrieved_texts": retrieved_texts
            },
            dynamic_config,
            output_keys=["messages"],
            stream_mode="values"
        )['messages']

        # Process and store AI responses
        for chunk in output:
            if isinstance(chunk, AIMessage):
                state_manager.append_chat_history(session_id, {
                    "role": "ai",
                    "content": chunk.content,
                    "timestamp": str(datetime.now())
                })

        logger.info(f"Successfully retrieved data for session: {session_id}")
        return state_manager.get_chat_history(session_id)

    except Exception as e:
        logger.error(f"Error in retrieve_embedded_data: {str(e)}")
        return None

# Initialize the embeddings and vector store
try:
    logger.info("Initializing embeddings")
    _ = state_manager.embeddings
    logger.info("Initializing vector store")
    _ = state_manager.vector_store
    logger.info("Successfully initialized application")
except Exception as e:
    logger.error(f"Failed to initialize application: {str(e)}")
    raise