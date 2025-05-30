from torch.optim.lr_scheduler import LRScheduler
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

# Add at top, before any other imports

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# Constants
VECTOR_STORE_DIR = "tafasir_quran_faiss_vectorstore"
VECTOR_STORE_PATH = Path(VECTOR_STORE_DIR).absolute()  # Use absolute path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "embeddings")
BATCH_SIZE = 32  # Add constant for batch control

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
    """Create embeddings with forced CPU usage."""
    if not os.path.exists(MODEL_PATH):
        raise ValueError(f"Model path does not exist: {MODEL_PATH}")

    torch.set_num_threads(4)
    device = "cpu"
    
    model_kwargs = {
        'device': device,
        'token': HF_TOKEN,
        'trust_remote_code': True
    }
    
    encode_kwargs = {
        'normalize_embeddings': True,
        'batch_size': 8,  # Reduced batch size further
        'max_length': 512,  # Increased max length
        'truncation': True,
        'padding': True
    }

    gc.collect()

    try:
        async with managed_executor() as executor:
            embeddings = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: HuggingFaceEmbeddings(
                        model_name=MODEL_PATH,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                    )
                ),
                timeout=120  # Increased timeout
            )
            gc.collect()
            # Test embedding
            test_result = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: embeddings.embed_query("test")
            )
            if not test_result or len(test_result) == 0:
                raise ValueError("Embedding initialization failed - empty test result")
            return embeddings
    except Exception as e:
        logger.error(f"Embedding creation failed: {str(e) or 'Unknown error during embedding creation'}")
        raise

class StateManager:
    """Memory-efficient state manager for embeddings and vector store."""
    def __init__(self):
        self._lock = asyncio.Lock()
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[FAISS] = None
        self._chat_history: Dict[str, List[dict]] = {}
        self._initializing_embeddings = False
        self._initializing_vector_store = False

    async def get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get embeddings with deadlock prevention"""
        if self._embeddings is not None:
            return self._embeddings
            
        if not self._initializing_embeddings:
            async with self._lock:
                if self._embeddings is None and not self._initializing_embeddings:
                    try:
                        self._initializing_embeddings = True
                        await self._initialize_embeddings()
                    finally:
                        self._initializing_embeddings = False
                        
        return self._embeddings

    async def get_vector_store(self) -> FAISS:
        """Re-initialize vector store only if it does not exist."""
        if self._vector_store is not None:
            print("Returning vector store")
            return self._vector_store

        if not self._initializing_vector_store:
            async with self._lock:
                if self._vector_store is None and not self._initializing_vector_store:
                    try:
                        
                        self._initializing_vector_store = True
                        await self._initialize_vector_store()
                    finally:
                        self._initializing_vector_store = False

        if self._vector_store is None:
            raise ValueError("Vector store initialization failed")

        return self._vector_store

    async def _initialize_embeddings(self):
        """Initialize embeddings asynchronously."""
        if self._embeddings is not None:
            return
            
        logger.info("Initializing new embeddings")
        try:
            self._embeddings = await create_embeddings()
            logger.info("Embeddings initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {str(e)}")
            self._embeddings = None  # Reset on failure
            raise

    async def _initialize_vector_store(self):
        """Initialize vector store only if it doesn't exist yet."""
        if self._vector_store is not None:
            return
        
        logger.info("Initializing vector store")
        try:
           

            # Define max_history at the beginning of the method
            max_history = 100  # Define maximum number of history entries
            
            # Acquire embeddings
            # ...existing code...
            # Use shared_executor for all blocking operations
            loop = asyncio.get_event_loop()
            if (VECTOR_STORE_PATH / "index.faiss").exists():
                logger.info("Loading existing vector store")
                self._vector_store = await loop.run_in_executor(
                    shared_executor,
                    lambda: FAISS.load_local(
                        folder_path=str(VECTOR_STORE_PATH),
                        embeddings=self._embeddings,
                        allow_dangerous_deserialization=True
                    )
                )
                logger.info(f"Loaded existing vector store with {len(self._vector_store.docstore._dict)} documents")
                
                # Enforce max_history limit
                if len(self._vector_store.docstore._dict) > max_history:
                    logger.info(f"Limiting vector store to the latest {max_history} documents")
                    # Sort documents by timestamp
                    sorted_docs = sorted(
                        self._vector_store.docstore._dict.items(),
                        key=lambda x: datetime.fromisoformat(x[1]['metadata']['timestamp']),
                        reverse=True
                    )
                    # Remove oldest documents beyond max_history
                    docs_to_remove = [doc_id for doc_id, _ in sorted_docs[max_history:]]
                    self._vector_store.remove_ids(docs_to_remove)
                    logger.info(f"Removed {len(docs_to_remove)} old documents from vector store")
                    # Save the updated vector store
                    await loop.run_in_executor(
                        shared_executor,
                        lambda: self._vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
                    )
                    logger.info("Updated vector store saved successfully")
            else:
                logger.info("Creating new vector store")
                sample_embedding = await loop.run_in_executor(
                    shared_executor,
                    lambda: self._embeddings.embed_query("test")
                )
                index = faiss.IndexFlatL2(len(sample_embedding))
                self._vector_store = FAISS(
                    embedding_function=self._embeddings,
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

            # Enforce max_history limit if newly created
            if len(self._vector_store.docstore._dict) > max_history:
                logger.info(f"Limiting newly created vector store to the latest {max_history} documents")
                sorted_docs = sorted(
                    self._vector_store.docstore._dict.items(),
                    key=lambda x: datetime.fromisoformat(x[1]['metadata']['timestamp']),
                    reverse=True
                )
                docs_to_remove = [doc_id for doc_id, _ in sorted_docs[max_history:]]
                self._vector_store.remove_ids(docs_to_remove)
                logger.info(f"Removed {len(docs_to_remove)} old documents from vector store")
                await loop.run_in_executor(
                    shared_executor,
                    lambda: self._vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
                )
                logger.info("Updated vector store saved successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            self._vector_store = None  # Reset on failure
            raise

    async def clear_cache(self):
        """Enhanced memory cleanup with lock protection"""
        if self._initializing_embeddings or self._initializing_vector_store:
            logger.warning("Cannot clear cache while initialization is in progress")
            return
            
        async with self._lock:
            try:
                # Clear embeddings
                if hasattr(self, '_embeddings'):
                    if hasattr(self._embeddings, 'client'):
                        del self._embeddings.client
                    del self._embeddings
                    self._embeddings = None

                # Clear vector store
                if hasattr(self, '_vector_store'):
                    if hasattr(self._vector_store, 'index'):
                        self._vector_store.index.reset()
                        del self._vector_store.index
                    del self._vector_store
                    self._vector_store = None

                # Clear history
                if hasattr(self, '_chat_history'):
                    self._chat_history.clear()

                # Force garbage collection
                gc.collect()
                gc.collect()  # Double collection to ensure references are cleared
                
                import sys
                sys.modules.pop('torch', None)  # Remove torch from sys.modules
                
                logger.info("Memory cache cleared completely")
                self._initializing_embeddings = False
                self._initializing_vector_store = False
            except Exception as e:
                logger.error(f"Error during cache clearing: {str(e)}")
                raise

    def get_chat_history(self, session_id: str) -> List[dict]:
        return self._chat_history.get(session_id, [])

    def append_chat_history(self, session_id: str, message: dict):
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(message)

def sanitize_text(text: str) -> str:
    """Clean and validate text input."""
    if not text:
        return ""
    # Remove duplicate end_of_turn markers
    text = text.replace("<end_of_turn><end_of_turn>", "<end_of_turn>")
    # Remove any trailing/leading whitespace
    return text.strip()

async def embed_data(message: str, ai_response: str, user_id: str) -> Dict[str, Any]:
    """Memory-optimized embedding with input validation"""
    try:
        print("Starting embed_data")
        # Input validation and sanitization
        if not message or not ai_response or not user_id:
            print(f"Missing parameters - message: {bool(message)}, ai_response: {bool(ai_response)}, user_id: {bool(user_id)}")
            return {
                "status": "error",
                "message": "Missing required parameters",
                "details": {
                    "message": bool(message),
                    "ai_response": bool(ai_response),
                    "user_id": bool(user_id)
                }
            }

        print("Sanitizing inputs")
        # Sanitize inputs
        message = sanitize_text(message)
        ai_response = sanitize_text(ai_response)
        print(f"Sanitized message length: {len(message)}")
        print(f"Sanitized response length: {len(ai_response)}")

        # Check content length
        if len(message) > 10000 or len(ai_response) > 10000:
            print("Content too long")
            return {
                "status": "error",
                "message": "Content too long",
                "details": {
                    "message_length": len(message),
                    "response_length": len(ai_response),
                    "max_length": 10000
                }
            }

        print("Getting vector store")
        vector_store = await state_manager.get_vector_store() # Remove the extra comma
        print("vector_store",vector_store)
        print("Got vector store")
        
        validate_inputs(message, user_id)
        print(f"Inputs validated for User: {user_id}")
        
        # Create timestamp once and reuse
        timestamp = datetime.now().isoformat()
        
        try:
            print("Creating document")
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
            print("Document created")

            # Generate unique ID with timestamp for ordering
            doc_id = f"document_{user_id}_{timestamp}"
            print(f"Generated doc_id: {doc_id}")

            try:
                print("Starting managed executor")
                async with managed_executor() as executor:
                    print("Adding document to vector store")
                    await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: vector_store.aadd_documents([new_document], ids=[doc_id])
                    )
                    print("Document added")
                    
                    print("Checking history limits")
                    max_history = 10
                    if len(vector_store.docstore._dict) > max_history:
                        print(f"Current store size: {len(vector_store.docstore._dict)}")
                        sorted_docs = sorted(
                            vector_store.docstore._dict.items(),
                            key=lambda x: datetime.fromisoformat(x[1]['metadata']['timestamp']),
                            reverse=True
                        )
                        docs_to_remove = [doc_id for doc_id, _ in sorted_docs[max_history:]]
                        vector_store.remove_ids(docs_to_remove)
                        print(f"Removed {len(docs_to_remove)} old documents")
                    
                    print("Saving vector store")
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            executor,
                            lambda: vector_store.save_local(folder_path=str(VECTOR_STORE_PATH))
                        ),
                        timeout=60
                    )
                    print("Vector store saved")
                    gc.collect()

                print("Operation completed successfully")
                return {
                    "status": "success",
                    "document_name": doc_id
                }
            except Exception as e:
                print(f"Error in executor block: {str(e)}")
                raise

        except Exception as e:
            print(f"Error in document processing: {str(e)}")
            e_msg = str(e) if str(e) else "Unknown error during document processing"
            return {
                "status": "error",
                "message": e_msg,
                "details": {
                    "stage": "document_processing",
                    "user_id": user_id
                }
            }

    except Exception as e:
        print(f"Error in main embed_data block: {str(e)}")
        e_msg = str(e) if str(e) else "Unknown embedding error occurred"
        return {
            "status": "error",
            "message": e_msg,
            "details": {
                "user_id": user_id,
                "message_length": len(message) if message else 0,
                "response_length": len(ai_response) if ai_response else 0
            }
        }

async def retrieve_embedded_data(message: Optional[str], user_id: str) -> Optional[List[dict]]:
    """Retrieve embedded data and manage chat history."""
    try:
        validate_inputs(message, user_id)
        logger.info(f"Retrieving data for User: {user_id}")

        vector_store = await asyncio.wait_for(
            state_manager.get_vector_store(), timeout=60  # Increased from 30
        )
        retrieved_context = await vector_store.asimilarity_search_with_score(message, k=1, filters={"user_id": user_id})  # Increased k to limit history
                
            
        docs_as_dicts = []
        for doc, score in retrieved_context:
            if score > 0.8:  # Only include results with a score > 0.8
                try:
                    print("doc ",doc)
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

        # Extract questions and responses with a maximum limit
        max_history = 1  # Define maximum number of history entries
        current_questions = [doc['metadata']['message'] for doc in sorted_docs[:max_history]]
        ai_responses = [doc['metadata']['Assistant']['response'] for doc in sorted_docs[:max_history]]

        # Format documents
        formatted_docs = []
        formatted_docs.append({
            "user_question": current_questions,
            "ai_response": ai_responses,
            "type": "history"
        })

        logger.info(f"Retrieved {len(formatted_docs)} documents for User {user_id}")
        gc.collect()
        return formatted_docs

    except asyncio.TimeoutError:
        logger.error("Retrieve embedded data operation timed out after 60 seconds")  # Updated from 30 seconds
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
    
    return result
     

# Ensure executor is properly shutdown on application exit
import atexit

@atexit.register
def shutdown_executor():
    executor.shutdown(wait=False)
    logger.info("Executor shutdown successfully")
