from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging

logger = logging.getLogger(__name__)

class MongoDBClient:
    client: AsyncIOMotorClient = None
    db = None

    @classmethod
    async def connect(cls):
        """Initialize the MongoDB client and database."""
        try:
            mongo_uri = os.getenv("MONGODB_URI")
            if not mongo_uri:
                raise ValueError("MONGODB_URI environment variable not set")
            cls.client = AsyncIOMotorClient(mongo_uri)
            cls.db = cls.client.get_default_database(default="Cluster0")
            logger.info("MongoDB connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    @classmethod
    def get_db(cls):
        """Retrieve the initialized database instance."""
        if cls.db is None:
            raise Exception("MongoDB client is not initialized. Call connect() first.")
        return cls.db

    @classmethod
    async def close(cls):
        """Close the MongoDB client connection."""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed")

# Add the initialize_db function
async def initialize_db():
    await MongoDBClient.connect()
