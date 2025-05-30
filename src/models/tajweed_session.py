from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class TajweedSession(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")  # Unique identifier for the session (MongoDB _id)
    user_id: str  # ID of the user who created the session
    session_id: str  # Unique session identifier for tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)  # Timestamp when the session was created
    updated_at: datetime = Field(default_factory=datetime.utcnow)  # Timestamp when the session was last updated
    tajweed_data: Dict  # Data related to Tajweed analysis, structured as needed
    summary: Optional[Dict] = None  # Add summary field for scores and metadata
