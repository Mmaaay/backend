from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.mongo_client import MongoDBClient
from constants import DB_CONNECTION_STRING
import logging
from faissEmbedding.embeddings_manager import state_manager
from db.supabase import Supabase
from constants import SUPABASE_URL, SUPABASE_KEY

logger = logging.getLogger(__name__)

class DatabaseLifespan:
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Initialize embeddings and vector store
            logger.info("Starting embeddings initialization")
            embeddings = state_manager.embeddings  # This will trigger initialization
            vector_store = state_manager.vector_store  # This will trigger initialization
            logger.info("Embeddings and Vector Store initialized successfully")
            
            # Initialize the supabase client
            supabase_instance = Supabase(SUPABASE_URL, SUPABASE_KEY)
            client = supabase_instance.get_client()
            app.state.supabase_client = client
            logger.info("Supabase client initialized successfully")
            
            # Initialize MongoDB connection
            await MongoDBClient.connect()
            app.state.database = MongoDBClient.get_db()
            app.state.quran_collection_user = app.state.database.get_collection("users")
            logger.info("Database connected and stored in app state")
            
            yield
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
        finally:
            # Cleanup
            await MongoDBClient.close()
            logger.info("MongoDB client closed")
            state_manager.clear_cache()
            logger.info("Embeddings cache cleared")