from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
from models.dto import ChatSessionBase, MessageContent, MessageRole
from models import Messages
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.chat_memory import (process_chat, 
                                      embed_message, retrieve_message)
from langchain.schema import HumanMessage  # Ensure HumanMessage is imported

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
    ):
        try:
            # Embed current message
            embed_result = await embed_message(content, session_id)
            if embed_result.get("status") == "error":
                logger.error(f"Embedding failed: {embed_result.get('message')}")
            print("backend embed_result", embed_result)
                
            # Retrieve context including history
            retrieved_content = await retrieve_message(content, session_id)
            retrieved_texts = "\n".join([msg["content"] for msg in retrieved_content]) if retrieved_content else ""
            
            # Get AI response with session history
            ai_content = await process_chat(
                {"content": content, "role": "human"},
                retrieved_texts,
                session_id=session_id
            )
            
            # Ensure AI response is clean
            ai_response = (
                ai_content.strip()
                if isinstance(ai_content, str)
                else str(ai_content)
            )
            
            # Build and store message
            message_data = {
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "is_visible": True,
                "metadata": {
                    "ai_response": ai_response,
                    "retrieved_content": retrieved_content,
                    "embedding_status": embed_result.get("status", "unknown")
                },
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            return await self.message_repo.create_message(message_data)
            
        except Exception as e:
            logger.error(f"Error in create_message: {str(e)}", exc_info=True)
            return {
                "Status": False,
                "Error": str(e)
            }
    
    async def get_session(
        self,
        session_id: str,
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_session(session_id)

