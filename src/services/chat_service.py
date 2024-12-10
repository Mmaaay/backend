from datetime import datetime
from typing import List, Optional
from models.dto import ChatSession
from models import Messages
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.app import embed_data , retrieve_embedded_data
class ChatService:
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.message_repo = MessageRepository()
    
    async def create_session(
        self,
        user_id: str,
        unique_chat_id:str,
        chat_title: Optional[str] = None
    ) -> ChatSession:
        return await self.chat_repo.create_session(
            user_id=user_id,
            chat_title=chat_title or "Untitled Session",
            unique_chat_id=unique_chat_id
        )
    
    async def create_message(
        self,
        session_id: str,
        user_id: str,
        content: Messages.MessageContent,
        role: str,
        metadata: Optional[dict] 
    ) -> Messages:
        embed_data(content.text , session_id)
        client_respone = retrieve_embedded_data(content.text , session_id)
        print(client_respone)
        message_data = {
            "session_id": session_id+user_id,
            "user_id": user_id,
            "role": "user",
            "content": {
                "text": content.text,
                "metadata": metadata or {}
            },
            "is_visible": True,
            "metadata":{
                "Content":client_respone[1]} 
        }
        return await self.message_repo.create_message(message_data)
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_chat_history(session_id, limit)
    
    async def reset_session(self, session_id: str) -> None:
        await self.message_repo.delete_session_messages(session_id)
        await self.chat_repo.delete(session_id)