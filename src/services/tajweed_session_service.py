from repos.tajweed_session_repository import TajweedSessionRepository
from utils.tajweed_score import extract_scores
from datetime import datetime

from typing import Optional

class TajweedSessionService:
    def __init__(self):
        self.repo = TajweedSessionRepository()

    async def create_session(self, user_id: str, session_id: str, tajweed_data: dict) -> str:
        if not user_id or not session_id:
            raise ValueError("User ID and Session ID must be provided.")
        if not tajweed_data:
            raise ValueError("Tajweed data must be provided.")
        now = datetime.utcnow()
        summary = extract_scores(
            tajweed_data,
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            updated_at=now
        )
        print(f"Summary: {summary}")
        return await self.repo.create(summary)

    def get_session(self, session_id: str):
        return self.repo.get_by_id(session_id)

    def update_session(self, session_id: str, update_data: dict) -> bool:
        update_data['updated_at'] = datetime.utcnow()
        return self.repo.update(session_id, update_data)

    def delete_session(self, session_id: str) -> bool:
        return self.repo.delete(session_id)
