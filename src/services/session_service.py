class SessionService:
    def __init__(self):
        self.sessions = {}  # In-memory session storage

    async def create_session(self, user_id: int, token: str):
        self.sessions[user_id] = token

    async def validate_session(self, user_id: int, token: str) -> bool:
        return self.sessions.get(user_id) == token

    async def delete_session(self, user_id: int):
        self.sessions.pop(user_id, None)

session_service = SessionService()