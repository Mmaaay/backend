from typing import List, Optional
from datetime import datetime
from models.dto import ChatSessionBase, Messages
from repos.base_repository import BaseRepository
from db.collections import Collections

class ChatRepository(BaseRepository[ChatSessionBase]):
    def __init__(self):
        super().__init__(Collections.chat_sessions(), ChatSessionBase)
    
    async def create_session(self, user_id: str, session_id: str , session_title:str) -> ChatSessionBase:
        session_data = {
            "user_id": user_id,
            "session_title": session_title,
            "session_id": session_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "is_active": True,
            "metadata": {}
        }
        user = await self.create(session_data)
        return user.user_id
    
    
    async def get_user_sessions(
        self, 
        user_id: str, 
        limit: int = 50,
        skip: int = 0
    ) -> List[ChatSessionBase]:
        cursor = self.collection.find(
            {"user_id": user_id}
        ).sort("created_at", -1).skip(skip).limit(limit)
        
        sessions = []
        async for session in cursor:
            session['_id'] = str(session['_id'])
            sessions.append(ChatSessionBase(**session))
        return sessions
    
    async def update_title(
        self, 
        session_id: str, 
        title: str
    ) -> Optional[ChatSessionBase]:
        update_data = {
            "title": title,
            "updated_at": datetime.now()
        }
        return await self.update(session_id, update_data)

class MessageRepository(BaseRepository[Messages]):
    def __init__(self):
        super().__init__(Collections.messages(), Messages)
        
    async def get_session(self , session_id:str) :
        return await self.collection.find_one({"session_id":session_id})
    
    async def create_message(self, message_data) :
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
    
  
        
        