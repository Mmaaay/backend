from fastapi import FastAPI, logger
from fastapi.middleware.cors import CORSMiddleware
import socketio
from faissEmbedding.embeddings_manager import initialize_services
from services.service_container import ServiceContainer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Socket.IO instance with updated settings
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=['*'],
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=1e8,
    allow_upgrades=True
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware with WebSocket support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize services using the container
try:
    services = ServiceContainer.initialize(sio)
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    raise

# Create Socket.IO app with updated configuration
socket_app = socketio.ASGIApp(
    socketio_server=sio,
    other_asgi_app=app,
    socketio_path='/socket.io/',
    static_files=None
)

# Use the socket_app as the main application
app = socket_app

# Add startup event to initialize services
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Socket.IO server...")
    await initialize_services()  # Ensure embeddings and vector store are initialized

# Optional: Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ...rest of your FastAPI routes...
