from datetime import datetime
from typing import List, Optional
from models.dto import ChatSessionBase, MessageContent, MessageRole
from models import Messages
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.embeddings_manager import embed_data , retrieve_embedded_data
class ChatService:
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.message_repo = MessageRepository()
    
    async def create_session(
        self,
        user_id: str,
        session_id:str,
        session_title: Optional[str] = None
    ) -> ChatSessionBase:
        return await self.chat_repo.create_session(
            user_id=user_id,
            session_title=session_title or "Untitled Session",
            session_id=session_id
        )
    
    
    
    async def create_message(
        self,
        session_id: str,
        user_id: str,
        content: str,
        role: MessageRole,
        metadata: Optional[dict] = None
    ) :
        # Embed data (assume these are working functions)
        embed_data(content, session_id)
        client_response = retrieve_embedded_data(content, session_id)
        print(client_response)

        # Build message data
        message_data = {
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "is_visible": True,
            "metadata": {
                "Content": client_response[-1] if client_response else {}
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return await self.message_repo.create_message(message_data)
    
    async def get_session(
        self,
        session_id: str,
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_session(session_id)
    
    