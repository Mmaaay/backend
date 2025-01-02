from datetime import datetime
from typing import List
from uuid import uuid4

import models.dto as dto
from constants import COOKIES_KEY_NAME
from fastapi import APIRouter, Depends, HTTPException, Request, status
from models import Messages
from services import jwt_service
from services.chat_service import ChatService
from utils.token import decode_token
from fastapi.responses import StreamingResponse
import logging

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
    token: str,
    chat_service: ChatService = Depends(get_chat_service)
) :
    # Decode token and ensure it's a dictionary
    user_data = decode_token(token)
    print(user_data)

    # Check if the 'id' exists in the dictionary
    if user_data.get("id") is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Update session details
    session.user_id = user_data["id"]
    session.session_id = user_data["id"]
    session.session_title = session.session_title or "Chat Session"

    # Create session
    session_id = await chat_service.create_session(session.user_id, session.session_id ,  session.session_title)
    return session_id


#send a message at the unique session id
@router.post("/send_message/{session_id}", status_code=status.HTTP_201_CREATED)
async def send_message(
    session_id: str,
    message: dto.MessageContent,
    chat_service: ChatService = Depends(get_chat_service)
):
    logger.info(f"Received message for session_id: {session_id} from user_id: {message.session_id}")
    
    # Create message stream
    async def event_generator():
        try:
            async for chunk in chat_service.create_message_stream(
                session_id=session_id,
                user_id=message.session_id or session_id,  # Ensure this is the correct user_id
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



