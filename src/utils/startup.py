from torch.optim.lr_scheduler import LRScheduler  # <-- Add this as the first import
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

            # Ensure HUGGINGFACE_API_KEY is set
            if not HUGGINGFACE_API_KEY:
                logger.error("HUGGINGFACE_API_KEY is not set in environment variables.")
                raise RuntimeError("HUGGINGFACE_API_KEY is required for Huggingface Hub login.")

            # Login to Huggingface Hub as the very first step
            logger.info("Logging into Huggingface Hub")
            try:
                login(token=HUGGINGFACE_API_KEY)
                logger.info("Huggingface Hub login successful")
            except Exception as e:
                logger.error(f"Failed to login to Huggingface Hub: {str(e)}")
                raise RuntimeError("Huggingface Hub login failed. Please check your HUGGINGFACE_API_KEY.") from e

            # Initialize core services
            logger.info("Starting service initialization")
            
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

            # Tajweed ASR model and rule trees initialization
            try:
                import torch
                import nemo.collections.asr as nemo_asr
                from utils.tajweed_utils import load_rule_trees

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                logger.info("Loading Tajweed ASR model...")
                asr_model = nemo_asr.models.ASRModel.from_pretrained(
                    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
                )
                asr_model.to(device)
                logger.info("Tajweed ASR model loaded.")

                logger.info("Loading Tajweed rule trees...")
                rule_trees = load_rule_trees()
                logger.info("Tajweed rule trees loaded.")

                app.state.tajweed_asr_model = asr_model
                app.state.tajweed_rule_trees = rule_trees
            except Exception as e:
                logger.error(f"Error initializing Tajweed model or rules: {str(e)}")
                app.state.tajweed_asr_model = None
                app.state.tajweed_rule_trees = None

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