from models.tajweed_session import TajweedSession
from db.collections import Collections
from typing import Optional
from .base_repository import BaseRepository

class TajweedSessionRepository(BaseRepository[TajweedSession]):
    def __init__(self):
        super().__init__(Collections.tajweed_sessions(), TajweedSession)

    async def create(self, session_summary: dict) -> str:
        # Accept and store only the summary dict
        if not session_summary.get('user_id') or not session_summary.get('session_id'):
            raise ValueError("User ID and Session ID must be provided.")
        data = session_summary.copy()
        if data.get("_id") is None:
            data.pop("_id", None)
        result = await self.collection.insert_one(data)
        return str(result.inserted_id)

    def get_by_id(self, session_id: str) -> Optional[dict]:
        data = self.collection.find_one({"session_id": session_id})
        if data:
            return data
        return None

    def update(self, session_id: str, update_data: dict) -> bool:
        result = self.collection.update_one({"session_id": session_id}, {"$set": update_data})
        return result.modified_count > 0

    def delete(self, session_id: str) -> bool:
        result = self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0
