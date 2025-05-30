from datetime import datetime, timedelta
from jose import jwt, JWTError
from constants import HASH_SALT, ALGORITHM
import logging

logger = logging.getLogger(__name__)

def create_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=7)  # Changed to 7 days default
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, HASH_SALT, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token.split(' ')[1]
            
        decoded = jwt.decode(token, HASH_SALT, algorithms=[ALGORITHM])
        return decoded
    except JWTError as e:
        logger.error(f"Failed to decode token: {str(e)}")
        logger.error(f"Token value: {token[:20]}...")
        return None
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {str(e)}")
        return None