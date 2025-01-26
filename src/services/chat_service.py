import logging
import asyncio
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from models.dto import ChatSessionBase, MessageContent, MessageRole , MessageDetails, MessageUserInterface
from models import Messages
from bson import ObjectId
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.chat_memory import (embed_system_response, process_chat, 
                                      embed_message, process_chat_stream, retrieve_message)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.message_repo = MessageRepository()
        
    async def get_user_sessions(self, user_id:str) -> List[MessageUserInterface]:
        #convert user_id to object id
        return await self.chat_repo.get_user_sessions(user_id)
    
    

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
            # 1. Save the user's current message
            if metadata is None:
                metadata = {}

            # 2. Retrieve all session history including the current message
            retrieved_content = await asyncio.wait_for(
                retrieve_message(content, user_id), timeout=30
            ) or []
            
             # ({
            #     "user_question": current_questions,
            #     "ai_response": ai_responses,
            #     "type": "history"
            # })
            
            ai_response_chunks = []
            
            # Extract related questions from history
            history_questions = [
                "{} \n".format([item.get('user_question') for item in retrieved_content])
            ]
            history_ai_responses = [
                "{} \n".format([item.get('ai_response') for item in retrieved_content])
            ]
            
            # # Current question to be processed
            current_question = content
            
            # # # # 3. Initiate AI response streaming
            async for chunk in process_chat_stream(
                messages=current_question,
                history_questions=history_questions,
                history_ai_responses=history_ai_responses,
                session_id=session_id
            ):
                logger.debug(f"Streaming chunk: {chunk}")
                ai_response_chunks.append(chunk)
                yield chunk
            # # # # 4. Combine chunks and embed the complete response
            ai_content = "".join(ai_response_chunks).strip()
            print(ai_content)
            if ai_content:
                full_message_data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": MessageRole.USER.value,
                    "content": current_question,
                    "is_deleted": False,
                    "created_at": datetime.now(),
                    "metadata": {
                        **metadata,
                        "ai_response": ai_content,
                        "role": MessageRole.ASSISTANT.value,
                        "timestamp": datetime.now().isoformat()
                    },
                }
                await asyncio.wait_for(
                    self.message_repo.create_message(full_message_data), timeout=30
                )
                
                embed_response_result = await embed_system_response(current_question, ai_content, user_id)
                if embed_response_result is None:
                    logger.error("embed_system_response returned None")
                elif embed_response_result.get("status") != "success":
                    logger.error(f"Failed to embed AI response: {embed_response_result.get('message')}")
                    
            else:
                full_message_data={
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": MessageRole.USER.value,
                    "content": current_question,
                    "is_deleted": False,
                    "created_at": datetime.now(),
                    "metadata": {
                        **metadata,
                        "ai_response": "Failed to generate response",
                        "role": MessageRole.USER.value,
                        "timestamp": datetime.now().isoformat()
                    },
                }
                await self.message_repo.create_message(full_message_data)
                embed_response_result = await embed_system_response(current_question, "Failed to generate response", user_id)
                if embed_response_result is None:
                    logger.error("embed_system_response returned None")
                elif embed_response_result.get("status") != "success":
                    logger.error(f"Failed to embed AI response: {embed_response_result.get('message')}")
                    

        except asyncio.TimeoutError:
            logger.error("Operation timed out after 30 seconds")
            yield "Error: Operation timed out."
        except Exception as e:
            logger.error(f"Error in create_message_stream: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
            
            
    async def get_session_chats(
        self,
        session_id: str,
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_chat_history(session_id)
    
    async def get_message_details(
        self,
        session_id: str
    ) -> List[Messages.Messages]:
        return await self.message_repo.get_message_details(session_id)




