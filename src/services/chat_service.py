from datetime import datetime
from typing import List, Optional, AsyncGenerator
import logging

logger = logging.getLogger(__name__)
from models.dto import ChatSessionBase, MessageContent, MessageRole
from models import Messages
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.chat_memory import (embed_system_response, process_chat, 
                                      embed_message, process_chat_stream, retrieve_message)
from langchain.schema import HumanMessage  # Ensure HumanMessage is imported
from server import sio  # Already imported

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
            
            # Retrieve all session history
            retrieved_content = await retrieve_message(None, session_id)
            print("retrieved content/ ", retrieved_content)
            
            ai_content = ""  # Initialize ai_content
            
            if retrieved_content:
                # Sort by timestamp to get messages in chronological order
                sorted_content = sorted(
                    retrieved_content,
                    key=lambda x: datetime.fromisoformat(x['metadata']['timestamp'])
                )
                
                # Get only the most recent question
                current_questions = [msg['content'] for msg in sorted_content[-1:]]
                history_questions = [msg['content'] for msg in sorted_content[:-1]]
                history = []  # No additional history needed
                
                # Get AI response with current questions and history
                ai_content = await process_chat(
                    current_questions,  # Most recent question
                    history_questions or [],            # Empty history
                    session_id=session_id
                )
            else:
                # Define a default AI response when there's no history
                ai_content = "I'm here to help with your questions about the Quran. How can I assist you today?"
            
            # **New Code: Ensure ai_content is not empty**
            if not ai_content:
                ai_content = "I'm here to help with your questions about the Quran. How can I assist you today?"
                logger.warning(f"'ai_content' was empty. Assigned default response for session: {session_id}")
            
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
    
    async def create_message_stream(
        self,
        session_id: str,
        user_id: str,
        content: str,
        role: MessageRole,
        metadata: Optional[dict] = None
    ) -> AsyncGenerator[str, None]:
        """Stream the message creation process and embed AI's response."""
        try:
            # Embed current message
            embed_result = await embed_message(content, session_id)
            if embed_result.get("status") == "error":
                logger.error(f"Embedding failed: {embed_result.get('message')}")
                yield "Error: Failed to embed message"
                return

            # Retrieve all session history
            retrieved_content = await retrieve_message(None, session_id)
            logger.debug(f"Retrieved content: {retrieved_content}")
            
            ai_content = ""
            ai_response_chunks = []

            if retrieved_content:
                # Sort by timestamp
                sorted_content = sorted(
                    retrieved_content,
                    key=lambda x: datetime.fromisoformat(x['metadata']['timestamp'])
                )
                
                # Get questions
                current_questions = [msg['content'] for msg in sorted_content[-1:]]
                history_questions = [msg['content'] for msg in sorted_content[:-1]]
                
                # Stream AI response
                async for chunk in process_chat_stream(
                    current_questions,
                    history_questions or [],
                    session_id=session_id
                ):
                    logger.debug(f"Streaming chunk: {chunk}")
                    ai_response_chunks.append(chunk)
                    yield chunk
                    # Emit each chunk to the client via Socket.IO
                    await sio.emit('ai_response', {'chunk': chunk}, room=session_id)
            else:
                default_response = "I'm here to help with your questions about the Quran. How can I assist you today?"
                logger.debug(f"No retrieved content. Sending default response: {default_response}")
                ai_response_chunks.append(default_response)
                yield default_response

            # Concatenate the streamed chunks to form the full AI response
            ai_content = "".join(ai_response_chunks).strip()
            logger.info(f"Complete AI response: {ai_content}")

            # Embed the AI's response
            embed_response_result = await embed_system_response(ai_content, session_id, content)
            if embed_response_result.get("status") != "success":
                logger.error(f"Failed to embed AI response: {embed_response_result.get('message')}")

        except Exception as e:
            logger.error(f"Error in create_message_stream: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"

    async def get_session(
        self,
        session_id: str,
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_session(session_id)
        return await self.message_repo.get_session(session_id)



