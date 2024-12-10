from pydantic import BaseModel, EmailStr, Field, field_validator

from datetime import datetime
from typing import Optional , Dict, List
from bson import ObjectId
from datetime import datetime


# USER
class CreateUser(BaseModel):
    name: str
    surname: str
    email: str
    role: str = "user"
    password: str = Field(..., min_length=4),
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class GetUser(BaseModel):
    id: str = Field(default=None , alias="_id")
    name: str
    surname: str
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
 

    
    
#chat


class ChatSessionBase(BaseModel):
    """Base model for chat session data"""
    chat_title: str = Field(..., description="Title of the chat session")
    unique_chat_id:str=Field(...,description="Unique Chat ID")
    user_id: str = Field(..., description="ID of the user who owns this session")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the session was created")
    updated_at: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp when the session was last updated")
    is_active: bool = Field(default=True, description="Whether the session is currently active")
    metadata: Dict = Field(default_factory=dict, description="Additional metadata for the session")

class CreateChatSession(ChatSessionBase):
    """Model for creating a new chat session"""
    pass

class UpdateChatSession(BaseModel):
    """Model for updating an existing chat session"""
    title: Optional[str] = Field(None, description="New title for the session")
    is_active: Optional[bool] = Field(None, description="Update the active status")
    metadata: Optional[Dict] = Field(None, description="Updated metadata")

class ChatSession(ChatSessionBase):
    """Model for a complete chat session with ID"""
    id: str = Field(..., alias="_id", description="Unique identifier for the session")

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChatSessionResponse(BaseModel):
    """Model for API responses containing chat session data"""
    session: ChatSession
    message_count: Optional[int] = Field(None, description="Number of messages in the session")
    last_message_at: Optional[datetime] = Field(None, description="Timestamp of the last message")

class ChatSessionList(BaseModel):
    """Model for listing multiple chat sessions"""
    sessions: List[ChatSession]
    total: int = Field(..., description="Total number of sessions")
    page: Optional[int] = Field(1, description="Current page number")
    page_size: Optional[int] = Field(50, description="Number of sessions per page")
    
    
#messages

from datetime import datetime
from typing import Optional, Dict, List, Union
from enum import Enum

class MessageRole(str, Enum):
    """Enum for different types of message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MessageContentType(str, Enum):
    """Enum for different types of message content"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    CODE = "code"
    MARKDOWN = "markdown"

class MessageContent(BaseModel):
    """Model for message content with support for different types"""
    text: str = Field(..., description="The actual message content")
    content_type: MessageContentType = Field(default=MessageContentType.TEXT, description="Type of the message content")
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
    content: MessageContent = Field(..., description="Content of the message")
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

class Messages(MessageBase):
    """Model for a complete message with ID"""
    id: str = Field(..., alias="_id", description="Unique identifier for the message")

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

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
    content_type_distribution: Dict[MessageContentType, int]
    created_at: datetime = Field(default_factory=datetime.now)