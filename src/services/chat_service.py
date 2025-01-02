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
    def __init__(self, socket_controller=None, db=None):
        if db is None:
            raise ValueError("Database instance is required for ChatService")
        self.chat_repo = ChatRepository(db)
        self.message_repo = MessageRepository(db)
        self.socket_controller = socket_controller
    
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
            
            # Retrieve all session history
            retrieved_content = await retrieve_message(None, session_id)
            retrieved_texts = []
            
            if retrieved_content:
                # Sort by timestamp to get messages in chronological order
                sorted_content = sorted(
                    retrieved_content,
                    key=lambda x: datetime.fromisoformat(x['metadata']['timestamp'])
                )
                
                # Extract all messages as current questions
                current_questions = [msg['content'] for msg in sorted_content]
                
                # Extract history as list of dictionaries (all messages except the first)
                history = sorted_content[1:] if len(sorted_content) > 1 else []
                
                # Get AI response with current questions and history
                ai_content = await process_chat(
                    current_questions,  # Current questions as list of strings
                    history,            # History as list of dictionaries
                    session_id=session_id,
                    socket_controller=self.socket_controller  # Pass socket controller
                )
            
            else:
                logger.warning(f"No retrieved_content for session_id: {session_id}. Using default context.")
                # Use the default context provided by retrieve_embedded_data
                ai_content = await process_chat(
                    [content],
                    retrieved_texts=[],
                    session_id=session_id,
                    socket_controller=self.socket_controller
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

