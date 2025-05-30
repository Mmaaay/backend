from enum import StrEnum
from typing import Optional
from pydantic import EmailStr, Field
from datetime import datetime
from models.Base import BaseDBModel

class User(BaseDBModel):
    name: str
    email: EmailStr
    password: str
    role: str
    
    class Role(StrEnum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"