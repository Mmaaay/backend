from typing import List, Optional
from datetime import datetime
from models.dto import ChatSession, Messages
from repos.base_repository import BaseRepository
from db.collections import Collections

class ChatRepository(BaseRepository[ChatSession]):
    def __init__(self):
        super().__init__(Collections.chat_sessions(), ChatSession)
    
    async def create_session(self, user_id: str, chat_title: str , unique_chat_id:str) -> ChatSession:
        session_data = {
            "user_id": user_id,
            "chat_title": chat_title,
            "unique_chat_id": unique_chat_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "is_active": True,
            "metadata": {}
        }
        return await self.create(session_data)
    
    async def get_user_sessions(
        self, 
        user_id: str, 
        limit: int = 50,
        skip: int = 0
    ) -> List[ChatSession]:
        cursor = self.collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        sessions = []
        async for session in cursor:
            session['_id'] = str(session['_id'])
            sessions.append(ChatSession(**session))
        return sessions
    
    async def update_title(
        self, 
        session_id: str, 
        title: str
    ) -> Optional[ChatSession]:
        update_data = {
            "title": title,
            "updated_at": datetime.now()
        }
        return await self.update(session_id, update_data)

class MessageRepository(BaseRepository[Messages]):
    def __init__(self):
        super().__init__(Collections.messages(), Messages)
    
    async def create_message(self, message_data: dict) -> Messages:
        message_data["created_at"] = datetime.now()
        message_data["updated_at"] = datetime.now()
        return await self.create(message_data)
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Messages]:
        cursor = self.collection.find(
            {"session_id": session_id}
        ).sort("created_at", 1).skip(skip).limit(limit)
        
        messages = []
        async for message in cursor:
            message['_id'] = str(message['_id'])
            messages.append(Messages(**message))
        return messages
    
    async def delete_session_messages(self, session_id: str) -> bool:
        try:
            result = await self.collection.delete_many({"session_id": session_id})
            return result.deleted_count > 0
        except:
            return False