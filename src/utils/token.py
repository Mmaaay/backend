from datetime import datetime, timedelta
import jwt
from constants import HASH_SALT, ALGORITHM
def create_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)  # Default expiry
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, HASH_SALT, algorithm=ALGORITHM)


def decode_token(token: str):
    try:
        return jwt.decode(token, HASH_SALT, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None