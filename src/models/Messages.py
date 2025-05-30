from datetime import datetime
from enum import StrEnum
from typing import Dict, Optional

from pydantic import Field

from models.Base import BaseDBModel

class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Messages(BaseDBModel):
    session_id: str
    user_id: str
    role: MessageRole
    content: str
    is_deleted: bool = False
    metadata: Optional[Dict] = Field(default_factory=dict)

class ChatSession(BaseDBModel):
    user_id: str
    session_id: str
    session_title: str
    is_active: bool = True