import logging
from datetime import datetime
from typing import Annotated, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse
import models.dto as dto
from constants import COOKIES_KEY_NAME
from models import Messages
from services.chat_service import ChatService
from utils.dependencies import get_current_user, user_dependency
from utils.token import decode_token

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

logger = logging.getLogger(__name__)

def get_chat_service():
    return ChatService()


@router.post("/create_session", status_code=status.HTTP_201_CREATED)
async def create_session(
    session: dto.createChatSessionResponse,
    current_user: user_dependency,
    chat_service: ChatService = Depends(get_chat_service)
) :
    # Use current_user instead of token
    session.user_id = current_user.id
    session.session_id = current_user.id
    session.session_title = session.session_title or "Chat Session"

    session_id = await chat_service.create_session(session.user_id, session.session_id, session.session_title)
    return session_id


#send a message at the unique session id
@router.post("/send_message/{session_id}", status_code=status.HTTP_201_CREATED)
async def send_message(
    session_id: str,
    message: dto.MessageContent,
    current_user: user_dependency,
    chat_service: ChatService = Depends(get_chat_service)
):
    # Verify user has access to this session
    if current_user.id is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")

    logger.info(f"Received message for session_id: {session_id} from user_id: {message.session_id}")
    
    # Create message stream
    async def event_generator():
        try:
            async for chunk in chat_service.create_message_stream(
                session_id=session_id,
                user_id= current_user.id,  # Ensure this is the correct user_id
                content=message.content,
                role=message.role
            ):
                logger.debug(f"Yielding chunk: {chunk}")
                yield chunk
        except Exception as e:
            logger.error(f"Error during event generation: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
    
    logger.info(f"Streaming response for session_id: {session_id}")
    return StreamingResponse(event_generator(), media_type="text/plain")

#get all chats for a specific session id
@router.get("/get_chat/{session_id}", response_model=List[dto.MessageUserInterface]) 
async def get_chat(
    session_id: str,
    current_user: user_dependency,
    chat_service: ChatService = Depends(get_chat_service)
) -> List[Messages.Messages]:
    # Verify user has access to this session
    chat = await chat_service.get_all_messages(session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if str(chat[0].user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")
    return chat

