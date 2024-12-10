from datetime import datetime
from typing import Optional
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator

class BaseDBModel(BaseModel):
    id: Optional[str] = Field(alias="_id", default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    @field_validator("id", mode="before")
    def validate_id(cls, value):
        if isinstance(value, ObjectId):
            return str(value)
        return value

    @field_validator("updated_at", mode="before")
    def update_timestamp(cls, value):
        return datetime.now()