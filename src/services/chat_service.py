import logging
import asyncio
from datetime import datetime
from typing import List, Optional, AsyncGenerator
from models.dto import ChatSessionBase, MessageContent, MessageRole , MessageDetails, MessageUserInterface
from models import Messages
from bson import ObjectId
from repos.chat_repository import ChatRepository, MessageRepository
from faissEmbedding.chat_memory import process_chat_stream
from faissEmbedding.embeddings_manager import embed_data, embed_system_response, retrieve_embedded_data

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
                retrieve_embedded_data(content, user_id), timeout=60  # Increased from 30
            ) or []
            ai_response_chunks = []  # Initialize within the method to reset per request
            
            # Extract related questions from history with a maximum limit
            max_history = 10  # Define maximum number of history entries
            history_questions = [
                "{} \n".format([item for sublist in [item.get('user_question') for item in retrieved_content[:max_history]] for item in sublist])
            ]
            history_ai_responses = [
                "{} \n".format([item for sublist in [item.get('ai_response') for item in retrieved_content[:max_history]] for item in sublist])
            ]
            
            # Current question to be processed
            current_question = content
            
            # 3. Initiate AI response streaming
            async for chunk in process_chat_stream(
                messages=current_question,
                history_questions=history_questions,
                history_ai_responses=history_ai_responses,
                session_id=session_id 
            ):
                logger.debug(f"Streaming chunk: {chunk}")
                ai_response_chunks.append(chunk)
                # Remove intermediate yields
            # Combine chunks and yield once
            temp_ai_content = "".join(ai_response_chunks).strip()
            yield temp_ai_content
            print("temp_ai_content", temp_ai_content)
            if temp_ai_content:
                # Create the user message before embedding
                full_user_message_data = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": MessageRole.USER.value,
                    "content": current_question,
                    "is_deleted": False,
                    "created_at": datetime.now(),
                    "metadata": {
                        **metadata,
                        "role": MessageRole.USER.value,
                        "timestamp": datetime.now().isoformat()
                    },
                }
                print("full_user_message_data", full_user_message_data)
                
                print("createing message")
                await asyncio.wait_for(
                    self.message_repo.create_message(full_user_message_data), timeout=60  # Increased from 30
                ) 
                print("message created")
                print("embeding data")
                embed_response_result = await embed_data(current_question, temp_ai_content, user_id)
                print("embed_response_result", embed_response_result)
                if embed_response_result is None:
                    logger.error("embed_system_response returned None for content: %s", temp_ai_content)
                elif embed_response_result.get("status") != "success":
                    logger.error("Failed to embed AI response: %s. Content: %s", 
                                 embed_response_result.get("message") or "No error message provided", temp_ai_content)

                # Release references
                del embed_response_result, full_user_message_data, temp_ai_content, ai_response_chunks
                temp_ai_content = None  # Clear the variable
                ai_response_chunks = []  # Reinitialize the list

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
                print("full_message_data", full_message_data)
                await self.message_repo.create_message(full_message_data)
                
                embed_response_result = await embed_system_response(current_question, "Failed to generate response", user_id)
                if embed_response_result is None:
                    logger.error("embed_system_response returned None for content: %s", "Failed to generate response")
                elif embed_response_result.get("status") != "success":
                    logger.error("Failed to embed AI response: %s. Content: %s", 
                                 embed_response_result.get("message") or "No error message provided", "Failed to generate response")
                
                # Release references
                del embed_response_result, full_message_data
                temp_ai_content = None  # Ensure temp_ai_content is cleared

        except asyncio.TimeoutError:
            logger.error("Operation timed out after 60 seconds")  # Updated message
            yield "Error: Operation timed out."
            
            # Create and embed the timeout error message
            error_message = "Operation timed out."
            full_error_message_data = {
                "session_id": session_id,
                "user_id": user_id,
                "role": MessageRole.ASSISTANT.value,
                "content": error_message,
                "is_deleted": False,
                "created_at": datetime.now(),
                "metadata": {
                    "role": MessageRole.ASSISTANT.value,
                    "timestamp": datetime.now().isoformat()
                },
            }
            await self.message_repo.create_message(full_error_message_data)
            await embed_data(content, error_message, user_id)
            del full_error_message_data, error_message

        except Exception as e:
            logger.error(f"Error in create_message_stream: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
            
            # Create and embed the generic error message
            error_message = f"Error: {str(e)}"
            full_error_message_data = {
                "session_id": session_id,
                "user_id": user_id,
                "role": MessageRole.ASSISTANT.value,
                "content": error_message,
                "is_deleted": False,
                "created_at": datetime.now(),
                "metadata": {
                    "role": MessageRole.ASSISTANT.value,
                    "timestamp": datetime.now().isoformat()
                },
            }
            await self.message_repo.create_message(full_error_message_data)
            await embed_system_response(content, error_message, user_id)
            del full_error_message_data, error_message
            
            
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









