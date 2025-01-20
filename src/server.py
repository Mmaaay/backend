import os
import sys
import uvicorn
from fastapi import FastAPI
import logging
from dotenv import load_dotenv
from utils.startup import DatabaseLifespan
from middleware.auth_middleware import AuthMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Environment variables loaded")

# Get the environment reload flag
ENV_RELOAD = os.getenv("ENV_RELOAD", "production").lower()

def create_app():
    app = FastAPI(lifespan=DatabaseLifespan.lifespan)
    
    # Add authentication middleware
    app.add_middleware(AuthMiddleware)
    
    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    # Register routes after app is fully initialized
    def register_routes():
        from controllers import auth_controller
        from controllers import chat_controller
        from controllers import supabase_controller
        
        app.include_router(auth_controller.router)
        app.include_router(chat_controller.router)
        app.include_router(supabase_controller.router)

    # Call route registration after all other setup is done
    register_routes()
    return app

# Initialize app at module level
app = create_app()

def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=3000, reload=False)
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")

if __name__ == "__main__":
    main()