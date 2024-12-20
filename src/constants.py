import dotenv
import logging
from os import getenv
from datetime import timedelta

dotenv.load_dotenv()

# Log the database connection string to check if it's loaded
DB_CONNECTION_STRING = getenv("MONGODB_URI")
COOKIES_KEY_NAME = "session_token"
SESSION_TIME = timedelta(days=30)
HASH_SALT = getenv("HASH_SALT", "qeqweasdshfhsfewfsdif")
COLLECTION_NAME = "Quran_Backend"
MONGODB_MAX_POOL_SIZE = 100
MONGODB_MIN_POOL_SIZE = 10