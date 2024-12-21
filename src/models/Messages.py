from datetime import datetime
from enum import StrEnum
from typing import Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, EmailStr, Field, field_validator

from models.Base import BaseDBModel

class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageContent(BaseModel):
    text: str
    metadata: Optional[Dict] = Field(default_factory=dict)


class Messages(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)  # Now directly a string
    session_id: str
    user_id: str
    role: MessageRole
    content: MessageContent
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_deleted: bool = False
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    # Validator for id field to convert ObjectId to str if needed
    @field_validator("id", mode="before")
    def validate_id(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return
class ChatSession(BaseDBModel):
    user_id: str
    sesssion_id:str
    session_title:str
    is_active: bool = True
   