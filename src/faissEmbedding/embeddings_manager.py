# embeddings_manager.py
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from threading import Lock

import faiss
import streamlit as st
import torch
from faissEmbedding.chat_memory import app, config
from langchain.schema import AIMessage, HumanMessage
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from st_pages import Page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VECTOR_STORE_PATH = "tafasir_quran_faiss_vectorstore"
MODEL_PATH = "sentence-transformers/all-MiniLM-L12-v2"

# Initialize environment and configurations
warnings.filterwarnings("ignore")

class StateManager:
    """Manages state for both FastAPI and Streamlit environments"""
    _instance = None
    _embeddings = None
    _vector_store = None
    _chat_history = {}
    _embeddings_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
        return cls._instance

    @property
    def embeddings(self):
        if self._embeddings is None:
            with self._embeddings_lock:
                if self._embeddings is not None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_kwargs = {'device': device}
                    encode_kwargs = {'normalize_embeddings': False}
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=MODEL_PATH,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs
                    )
            logger.info(f"Initialized new embeddings on device: {device}")
        else:
            logger.info("Embeddings already initialized, skipping re-load.")
        return self._embeddings


    @property
    def vector_store(self):
        if self._vector_store is None:
            index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
            self._vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            logger.info("Initialized new vector store")
        return self._vector_store

    def get_chat_history(self, session_id: str) -> list:
        return self._chat_history.get(session_id, [])

    def append_chat_history(self, session_id: str, message: dict):
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(message)

    def clear_cache(self):
        self._embeddings = None
        self._vector_store = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleared embeddings cache and CUDA memory")

# Create a singleton instance
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