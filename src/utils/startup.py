from fastapi import FastAPI
from pymongo import MongoClient
from db.mongo_client import MongoDBClient
from constants import DB_CONNECTION_STRING, COLLECTION_NAME
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

if not DB_CONNECTION_STRING:
    raise Exception("DB connection string not provided")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     client = MongoClient(host=DB_CONNECTION_STRING)
#     database = client['Quran']

#     # Ping the database to check the connection
#     pong = database.command("ping")
    
#     if int(pong["ok"]) != 1:   
#         client.close()
#         raise Exception("Cluster connection is not okay!")

#     # Store the database and collection in the app state
#     app.state.database = database
#     app.state.quran_collection_user = database.get_collection(COLLECTION_NAME)
    
#     yield  # This allows the application to run
    
#     client.close()


class DatabaseLifespan:
    @staticmethod
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            # Initialize all database connections
            await MongoDBClient.connect()
            # Add any other initialization here (Redis, Elasticsearch, etc.)
            
            yield  # Application runs here
            
        finally:
            # Cleanup all connections
            await MongoDBClient.close()
            # Add any other cleanup here