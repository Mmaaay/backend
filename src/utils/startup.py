from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.mongo_client import MongoDBClient
from constants import DB_CONNECTION_STRING
import logging
from faissEmbedding.embeddings_manager import state_manager  # Import the state manager
from db.supabase import Supabase
from constants import SUPABASE_URL, SUPABASE_KEY  # Import constants


logger = logging.getLogger(__name__)

class DatabaseLifespan:
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            
            _ = state_manager.embeddings
            _ = state_manager.vector_store
            logger.info("Embeddings and Vector Store initialized successfully.")
            
            # Initialize the supabase client
            supabase_instance = Supabase(SUPABASE_URL, SUPABASE_KEY)  # Create an instance
            client = supabase_instance.get_client()
            app.state.supabase_client = client
            logger.info("Supabase client initialized successfully.")
            
            # Initialize MongoDB connection
            await MongoDBClient.connect()

            # Store the database in app state
            app.state.database = MongoDBClient.get_db()
            app.state.quran_collection_user = app.state.database.get_collection("users")

            logger.info("Database connected and stored in app state")

            yield  # Application is running here

        except Exception as e:
            logger.error(f"Error during database connection: {e}")
            raise

        finally:
            # Cleanup MongoDB client
            await MongoDBClient.close()
            logger.info("MongoDB client closed")
