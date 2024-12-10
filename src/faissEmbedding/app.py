from langchain_community.vectorstores import FAISS
import faiss
from st_pages import Page
import streamlit as st
import os
import warnings
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import HumanMessage, AIMessage
from langchain_core.documents import Document
from pathlib import Path

import torch
from faissEmbedding.chat_memory import app, config

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
os.environ["GOOGLE_API_KEY"] = 'AIzaSyDu6JN_L9gojotvFa8ALFgYO3mux9eB3-U'

def init_embeddings() -> HuggingFaceEmbeddings:
    """Initialize and return HuggingFace embeddings."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {'device': device}  # You might want to add cuda availability check
        encode_kwargs = {'normalize_embeddings': False}
        return HuggingFaceEmbeddings(
            model_name=MODEL_PATH,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}")
        raise

def init_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Initialize and return FAISS vector store."""
    try:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        return FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {str(e)}")
        raise

def validate_inputs(message: str, session_id: str) -> None:
    """Validate input parameters."""
    if not message or not isinstance(message, str):
        raise ValueError("Message must be a non-empty string")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Session ID must be a non-empty string")

# Initialize global variables
try:
    HF_embeddings = init_embeddings()
    vector_store = init_vector_store(HF_embeddings)
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    raise

def embed_data(message: str, session_id: str) -> Dict[str, Any]:
    """
    Embed data into vector store with error handling.
    Returns dict with status and additional information.
    """
    try:
        validate_inputs(message, session_id)
        logger.info(f"Embedding data for session: {session_id}")

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
            embeddings=HF_embeddings,
            allow_dangerous_deserialization=True
        ) if Path(f"{VECTOR_STORE_PATH}/index.faiss").exists() else vector_store

        retriever = faiss_store.as_retriever()

        # Initialize session state if needed
        if session_id not in st.session_state:
            st.session_state[session_id] = []

        # Add user message to chat history
        st.session_state[session_id].append({
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
                st.session_state[session_id].append({
                    "role": "ai",
                    "content": chunk.content,
                    "timestamp": str(datetime.now())
                })

        logger.info(f"Successfully retrieved data for session: {session_id}")
        print(st.session_state[session_id])
        return st.session_state[session_id]

    except Exception as e:
        logger.error(f"Error in retrieve_embedded_data: {str(e)}")
        return None

# Example usage:
# result = embed_data("Hello", "session_123")
# chat_history = retrieve_embedded_data("Query", "session_123")