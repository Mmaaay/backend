from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.mongo_client import MongoDBClient
import logging
from faissEmbedding.embeddings_manager import state_manager, initialize_services
from db.supabase import Supabase
from constants import SUPABASE_URL, SUPABASE_KEY
import asyncio

logger = logging.getLogger(__name__)

class DatabaseLifespan:
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Initialize core services
            logger.info("Starting service initialization")
            
            # Initialize MongoDB first (lightweight)
            await MongoDBClient.connect()
            app.state.database = MongoDBClient.get_db()
            app.state.quran_collection_user = app.state.database.get_collection("users")
            logger.info("MongoDB connected successfully")
            
            # Initialize Supabase (lightweight)
            supabase_instance = Supabase(SUPABASE_URL, SUPABASE_KEY)
            client = supabase_instance.get_client()
            app.state.supabase_client = client
            logger.info("Supabase client initialized successfully")
            
            # Initialize embeddings and vector store
            try:
                await initialize_services()
                logger.info("Embeddings and Vector Store initialized successfully.")
            except Exception as e:
                logger.error(f"Error initializing embeddings: {str(e)}")
            except MemoryError:
                logger.warning("Insufficient memory for ML services - will initialize on demand")
            except Exception as e:
                logger.error(f"Error initializing ML services: {str(e)}")
                # Continue anyway - services will initialize on demand
            
            yield
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
        finally:
            # Cleanup in reverse order
            await state_manager.clear_cache()
            await MongoDBClient.close()
            logger.info("Cleanup completed")