from enum import StrEnum
from typing import Optional
from bson import ObjectId
from pydantic import BaseModel, Field, EmailStr, field_validator
from datetime import datetime

class User(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)  # Now directly a string
    name: str
    imageUrl : str
    email: EmailStr
    password: str
    role: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    class Role(StrEnum):
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"
    
    # Validator for id field to convert ObjectId to str if needed
    @field_validator("id", mode="before")
    def validate_id(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value
     
    @field_validator("updated_at", mode="before")
    def validate_updated_at(cls, value):
        return datetime.now()