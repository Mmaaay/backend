import os
import sys
import uvicorn
from constants import DB_CONNECTION_STRING
from controllers import auth_controller, chat_controller, supabase_controller
from fastapi import Depends, FastAPI
from pymongo import MongoClient
from utils import startup
import logging
from dotenv import load_dotenv
from db.supabase import Supabase
import socketio  # Added import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "on", "yes"}
sio = socketio.AsyncServer(async_mode='asgi')  # Initialize SocketIO
app = FastAPI(lifespan=startup.DatabaseLifespan.lifespan)
sio_app = socketio.ASGIApp(sio, app)  # Wrap FastAPI with SocketIO

app.include_router(auth_controller.router)
app.include_router(chat_controller.router)
app.include_router(supabase_controller.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    # Join room based on session_id if needed
    # Example: await sio.emit('welcome', {'message': 'Connected to server'}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")

@sio.event
async def message(sid, data):
    logger.info(f"Message from {sid}: {data}")
    await sio.emit('response', {'message': 'Message received'}, to=sid)

# Modify the main function to run sio_app instead of app
def main(argv=sys.argv[1:]):
    try:
        uvicorn.run(sio_app, host="0.0.0.0", port=3000, reload=False)
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")

if __name__ == "__main__":
    main()

# Listening for changes is handled in chat_service.py where AI responses are emitted to rooms identified by session_id