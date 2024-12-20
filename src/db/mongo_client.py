from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from contextlib import asynccontextmanager
from constants import DB_CONNECTION_STRING
import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MongoDBClient:
    client: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    async def connect(cls):
        """Initialize database connection"""
        if cls.client is None:
            try:
                cls.client = AsyncIOMotorClient(DB_CONNECTION_STRING)
                logger.info("MongoDB connection established")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise

    @classmethod
    async def close(cls):
        """Close database connection"""
        if cls.client is not None:
            try:
                cls.client.close()
                cls.client = None
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")

    @classmethod
    def get_db(cls):
        """Get database instance"""
        if cls.client is None:
            raise Exception("Database not initialized")
        return cls.client["Quran"]  # Replace with a configurable database name if needed
    
    @classmethod
    @asynccontextmanager
    async def get_session(cls):
        """Provide a MongoDB session"""
        if cls.client is None:
            raise Exception("MongoDB client is not initialized.")
        async with cls.client.start_session() as session:
            try:
                yield session
            except Exception as e:
                logger.error(f"Error with MongoDB session: {e}")
                raise
