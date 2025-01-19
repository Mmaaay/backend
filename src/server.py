import os
import sys
import uvicorn
from constants import DB_CONNECTION_STRING
from controllers import auth_controller, supabase_controller
from fastapi import FastAPI
from utils import startup
import logging
from dotenv import load_dotenv
import socketio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "on", "yes"}
sio = socketio.AsyncServer(async_mode='asgi')
app = FastAPI(lifespan=startup.DatabaseLifespan.lifespan)
sio_app = socketio.ASGIApp(sio, app)

# Initialize controllers
from controllers.socket_controller import SocketController
socket_controller = SocketController(sio)

# Import routers after app initialization
from controllers.router import chat_router

# Include routers
app.include_router(auth_controller.router)
app.include_router(supabase_controller.router)
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def main(argv=sys.argv[1:]):
    try:
        uvicorn.run(sio_app, host="0.0.0.0", port=3000, reload=False)
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")

if __name__ == "__main__":
    main()