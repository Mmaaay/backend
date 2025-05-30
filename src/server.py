import os
from torch.optim.lr_scheduler import LRScheduler
import sys
import uvicorn
from fastapi import FastAPI
import logging
from dotenv import load_dotenv
from middleware.auth_middleware import AuthMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Environment variables loaded")

# Get the environment reload flag
ENV_RELOAD = os.getenv("ENV_RELOAD", "production").lower()

def create_app():
    app = FastAPI()

    # Add authentication middleware
    app.add_middleware(AuthMiddleware)

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    from contextlib import asynccontextmanager
    from utils.startup import DatabaseLifespan

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with DatabaseLifespan.lifespan(app):
            # Register routers after all initialization is complete
            from controllers.auth_controller import router as auth_router
            from controllers.chat_controller import router as chat_router
            from controllers.supabase_controller import router as supabase_router
            from controllers.tajweed_controller import router as tajweed_router

            app.include_router(auth_router)
            app.include_router(chat_router)
            app.include_router(supabase_router)
            app.include_router(tajweed_router)
            yield

    app.router.lifespan_context = lifespan
    return app

# Initialize app at module level
app = create_app()

def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=os.getenv("ENV_RELOAD", "production").lower() == "development", log_level="info")
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")

if __name__ == "__main__":
    main()