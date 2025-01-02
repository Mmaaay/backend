from fastapi import FastAPI
from contextlib import asynccontextmanager
from controllers import auth_controller, supabase_controller
from faissEmbedding.chat_memory import process_chat
from db.mongo_client import MongoDBClient
import logging
from faissEmbedding.embeddings_manager import retrieve_embedded_data, state_manager, initialize_services
from db.supabase import Supabase
from constants import SUPABASE_URL, SUPABASE_KEY
import asyncio
import socketio
from controllers.router import chat_router  # Import routers here

# Initialize Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')

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

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=DatabaseLifespan.lifespan)

# Include routers directly on the FastAPI app
app.include_router(auth_controller.router)
app.include_router(supabase_controller.router)
app.include_router(chat_router)

# Wrap the FastAPI app with Socket.IO ASGI app
app_asgi = socketio.ASGIApp(sio, app)

# Attach event handlers outside the lifespan context manager
@sio.event
async def message(sid, data):
    logger.info(f"Received message from {sid}: {data}")
    session_id = data.get("session_id")
    user_id = data.get("user_id")
    content = data.get("content")
    role = data.get("role")
    
    if role != "human":
        logger.warning("Invalid role.")
        await sio.emit("error", {"message": "Invalid role."}, to=sid)
        return
    
    # Process the message using embeddings_manager and pass sio and sid
    response = await retrieve_embedded_data(content, session_id)
    
    if response:
        # Optionally emit an initial acknowledgment
        await sio.emit("response", {"session_id": session_id, "response": "Processing your request...", "is_end": False}, to=sid)
        
        # Start streaming AI response
        await process_chat(
            messages=[content],
            retrieved_texts=response,
            session_id=session_id,
            sio=sio,
            sid=sid
        )
    else:
        await sio.emit("error", {"message": "Failed to retrieve data."}, to=sid)

# Ensure FastAPI is served via the Socket.IO ASGI App
# This typically involves configuring your ASGI server (e.g., Uvicorn) to run `app_asgi` instead of `app`
# At the end of the file, ensure that the ASGI app is exposed correctly
# This typically involves removing or modifying the existing app declaration
# and ensuring that your ASGI server points to `app_asgi`

# Example modification if using Uvicorn:
# Instead of running `app`, run `app_asgi` as the ASGI application.

# If using a separate run script, ensure it references `app_asgi`
# Example run command:
# uvicorn src.utils.startup:app_asgi --host 0.0.0.0 --port 3000

# Remove any existing `if __name__ == '__main__':` blocks related to FastAPI