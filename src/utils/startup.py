from fastapi import FastAPI
from contextlib import asynccontextmanager
from db.mongo_client import MongoDBClient
import logging
from faissEmbedding.embeddings_manager import state_manager, initialize_services
from db.supabase import Supabase
from huggingface_hub import login
from constants import SUPABASE_URL, SUPABASE_KEY, HUGGINGFACE_API_KEY
import asyncio
import os
import gc

logger = logging.getLogger(__name__)

class DatabaseLifespan:
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            # Force CPU mode before any ML initialization
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['USE_TORCH'] = 'cpu'
            os.environ['FORCE_CPU'] = '1'
            
            # Initialize core services
            logger.info("Starting service initialization")
            
            # Initialize MongoDB first (lightweight)
            await MongoDBClient.connect()
            app.state.database = MongoDBClient.get_db()
            app.state.quran_collection_user = app.state.database.get_collection("users")
            logger.info("MongoDB connected successfully")
            
            logger.info("Logging into Huggingface Hub")
            login(token=HUGGINGFACE_API_KEY)
            logger.info("Huggingface Hub login successful")
            
            
            # Initialize Supabase (lightweight)
            supabase_instance = Supabase(SUPABASE_URL, SUPABASE_KEY)
            client = supabase_instance.get_client()
            app.state.supabase_client = client
            logger.info("Supabase client initialized successfully")
            
            # Initialize embeddings and vector store
            try:
                await initialize_services()
                logger.info("Embeddings and Vector Store initialized successfully on CPU")
            except MemoryError:
                logger.error("Memory error during initialization")
                await state_manager.clear_cache()
                raise
            except Exception as e:
                logger.error(f"Error initializing ML services: {str(e)}")
                await state_manager.clear_cache()
                raise
            
            yield
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise
        finally:
            # Enhanced cleanup
            await state_manager.clear_cache()
            await MongoDBClient.close()
            gc.collect()
            gc.collect()
            logger.info("Cleanup completed")