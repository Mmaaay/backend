import os
import sys
import uvicorn
from constants import DB_CONNECTION_STRING
from controllers import auth_controller, chat_controller
from fastapi import Depends, FastAPI
from pymongo import MongoClient
from utils import startup
import logging




DEBUG = os.environ.get("DEBUG", "").strip().lower() in {"1", "true", "on", "yes"}
app = FastAPI(lifespan=startup.DatabaseLifespan.lifespan )

app.include_router(auth_controller.router)
app.include_router(chat_controller.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

def main(argv=sys.argv[1:]):
    try:
        uvicorn.run("server:app", host="0.0.0.0", port=80, reload=DEBUG)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()