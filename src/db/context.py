from typing import Generator, Any
from pymongo import MongoClient
from constants import DB_CONNECTION_STRING


if not DB_CONNECTION_STRING:
    raise Exception("DB connection string not provided")

client = MongoClient(DB_CONNECTION_STRING)
db_name ="Quran"
db = client[db_name]
users_collection = db['users']  # Assuming 'users' is the collection name
chat_sessions_collection = db['chat_sessions']  # Assuming 'chat_sessions' is the collection name

def create_db() -> None:
    """
    MongoDB is schemaless, so there's no need to create tables or collections ahead of time.
    You can use this function to ensure the connection is valid.
    """
    client.admin.command('ping')


def get_db() -> Generator[Any, None, None]:
    """
    Yields the MongoDB database connection for interactions within the current request context.
    """
    yield db


def auto_create_db():
    """
    Automatically connects to MongoDB. If the connection fails, the database is created
    and the function ensures that the connection is established.
    """
    try:
        create_db()  # Ping to check connection
    except Exception:
        raise Exception(f"Failed to connect to database: {db_name}")
