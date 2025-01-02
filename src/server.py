import os
import sys
import uvicorn
from faissEmbedding.embeddings_manager import initialize_services
from constants import DB_CONNECTION_STRING
from controllers import auth_controller, supabase_controller
from fastapi import FastAPI
from utils import startup
import logging
from dotenv import load_dotenv
import socketio
from services.service_container import ServiceContainer
from db.mongo_client import initialize_db  # Ensure this import exists and is correct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "on", "yes"}

# Initialize Socket.IO server from startup.py
from utils.startup import app_asgi  # Import the ASGI app with Socket.IO

# Remove references to app_asgi.app and include routers in startup.py instead

if __name__ == '__main__':
    uvicorn.run("utils.startup:app_asgi", host="0.0.0.0", port=3000, reload=True)