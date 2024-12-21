import os
import sys
import uvicorn
from constants import DB_CONNECTION_STRING
from controllers import auth_controller, chat_controller,supabase_controller
from fastapi import Depends, FastAPI
from pymongo import MongoClient
from utils import startup
import logging
from dotenv import load_dotenv
from db.supabase import Supabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "on", "yes"}
app = FastAPI(lifespan=startup.DatabaseLifespan.lifespan )



app.include_router(auth_controller.router)
app.include_router(chat_controller.router)
app.include_router(supabase_controller.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", port=80, host="0.0.0.0", reload=DEBUG)
    except Exception as e:
        logger.error(f"Failed to start the server: {str(e)}")
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")

if __name__ == "__main__":
    main()