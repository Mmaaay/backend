from pydantic import BaseModel, EmailStr, Field, field_validator
from datetime import datetime
from typing import Any, Optional , Dict, List
from bson import ObjectId
from datetime import datetime
from models.Messages import MessageRole, Messages
from models.Base import BaseDBModel


# ==========================
# USER MODELS
# ==========================
# USER
class CreateUser(BaseModel):
    name: str
    email: str
    role: str = "user"
    password: str = Field(..., min_length=4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class GetUser(BaseModel):
    id: str = Field(default=None , alias="_id")
    name: str
    role: str
    email: str
    hashed_password: str
    updated_at: datetime
    created_at: datetime      
    
       
    
class UpdateUser(BaseModel):
    name: str
    surname: str
    
class LoginUser(BaseModel):
    email: str
    password: str
    
class UpdateUserPass(BaseModel):
    email: str
    new_password: str = Field(..., min_length=4)
    old_password: str = Field(..., min_length=4)
    
    
class changeUserPass(BaseModel):
    new_password: str = Field(... , min_length=4)
    
# ==========================
# TOKEN MODELS
# ==========================
# Token
class Token(BaseModel):
    user_id: str   
    role : str
    exp: datetime
    
    @field_validator("user_id", mode="before")
    def validate_id(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value
 

    
    
# ==========================
# CHAT SESSION MODELS
# ==========================
#chat
class ChatSessionBase(BaseDBModel):
    """Base model for chat session data"""
    session_title: str = Field(..., description="Title of the chat session")
    session_id:str=Field(...,description="Unique Chat ID")
    user_id: str = Field(..., description="ID of the user who owns this session")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the session was created")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the session was last updated")
    is_active: bool = Field(default=True, description="Whether the session is currently active")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the session")

class CreateChatSession(ChatSessionBase):
    """Model for creating a new chat session"""
    user_id : str = Field(..., description="ID of the user who owns this session")
    session_id : str = Field(..., description="ID of the session")
    session_title : str = Field(..., description="Title of the session")
    pass

class createChatSessionResponse(BaseModel):
    """Model for creating a new chat session"""
    session_title : str = Field(..., description="Title of the session")
    pass

class CreateMessage(BaseModel):
    """Model for creating a new message"""
    session_id: str = Field(..., description="ID of the chat session this message belongs to")
    user_id: str = Field(..., description="ID of the user who sent the message")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the message")

# ==========================
# MESSAGE MODELS
# ==========================
from datetime import datetime
from typing import Optional, Dict, List, Union
from enum import Enum

class MessageRole(str, Enum):
    """Enum for sender roles"""
    USER = "user"
    ASSISTANT = "assistant"

 

class MessageContent(BaseModel):
    """Model for message content with support for different types"""
    session_id: str = Field(..., description="ID of the chat session this message belongs to")
    content: str = Field(..., description="The actual message content")
    role: MessageRole = Field(..., description="Role of the message sender")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the content")
    
    @field_validator('metadata')
    def validate_metadata(cls, v):
        """Ensure metadata is properly formatted"""
        if v is None:
            return {}
        return v

class MessageBase(BaseModel):
    """Base model for message data"""
    session_id: str = Field(..., description="ID of the chat session this message belongs to")
    user_id: str = Field(..., description="ID of the user who sent the message")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the message was created")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the message was last updated")
    parent_id: Optional[str] = Field(None, description="ID of the parent message if this is a reply")
    is_visible: bool = Field(default=True, description="Whether the message is visible in the chat")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the message")

class CreateMessage(MessageBase):
    """Model for creating a new message"""
    pass

class UpdateMessage(BaseModel):
    """Model for updating an existing message"""
    content: Optional[MessageContent] = Field(None, description="Updated content of the message")
    is_visible: Optional[bool] = Field(None, description="Update visibility status")
    metadata: Optional[Dict] = Field(None, description="Updated metadata")

class Messages(BaseModel):
    session_id: str
    content: str
    role: MessageRole
    metadata: Dict[str , Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class MessageResponse(BaseModel):
    """Model for API responses containing message data"""
    usr_message: str
    message: str

class MessageList(BaseModel):
    """Model for listing multiple messages"""
    messages: List[Messages]
    total: int = Field(..., description="Total number of messages")
    has_more: bool = Field(default=False, description="Whether there are more messages to load")
    next_cursor: Optional[str] = Field(None, description="Cursor for pagination")

class MessageThread(BaseModel):
    """Model for representing a message thread"""
    parent_message: Messages
    replies: List[Messages]
    total_replies: int
    metadata: Dict = Field(default_factory=dict)

class MessageAnalytics(BaseModel):
    """Model for message analytics data"""
    total_messages: int
    user_message_count: int
    assistant_message_count: int
    average_response_time: Optional[float]
    session_duration: Optional[float]
    WER : Optional[float] = Field(None, description="Word Error Rate")
    created_at: datetime = Field(default_factory=datetime.now)
    
    
class MessageUserInterface(BaseModel):
    """Model for message data in the user interface"""
    session_id: str
    user_id: str
    session_title: str
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
    
class MessageDetails(BaseModel):
    """Model for message details in the user interface"""
    session_id: str
    user_id: str
    role: str
    content: List[str]
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
    


class GetUserSessionsRequest(BaseModel):
    pass
class GetUserMessagesRequest(BaseModel):
    session_id: str

# ==========================
# TAJWEED ANALYSIS MODELS
# ==========================
# Tajweed analysis models

class TajweedRule(BaseModel):
    rule: str
    start_idx: int
    end_idx: int
    text: str
    start_time: float
    end_time: float
    spans_words: Optional[bool] = False

class WordAnalysis(BaseModel):
    word: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    tajweed_rules: List[TajweedRule] = Field(default_factory=list)