from asyncio.log import logger
from typing import Annotated
from datetime import datetime

from fastapi import Depends
from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi import HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError , jwt

from constants import COOKIES_KEY_NAME
from repos import user_repository as db
from services.user_service import UserService   
from constants import HASH_SALT, ALGORITHM
from utils.token import decode_token as token_decode


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> db.User:
    user_repo = db.UserRepository()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        raise credentials_exception
        
    payload = token_decode(token)
    if not payload:
        logger.error("Token decode failed")
        raise credentials_exception
    
    # Check token expiration
    exp = payload.get("exp")
    if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user_id: str = payload.get("id")
    if user_id is None:
        raise credentials_exception

    user = await user_repo.get_by_id(user_id)
    if user is None:
        raise credentials_exception
    return user



user_dependency = Annotated[db.User, Depends(get_current_user)]