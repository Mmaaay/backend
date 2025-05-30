import logging
from datetime import datetime
from typing import Annotated, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from fastapi.responses import StreamingResponse
from models.dto import createChatSessionResponse, MessageContent, MessageDetails, MessageUserInterface, GetUserSessionsRequest, GetUserMessagesRequest
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
    session: createChatSessionResponse,
    chat_service: ChatService = Depends(get_chat_service),
    current_user = Depends(get_current_user)
) :
    # Verify user has access to this session
    
    if current_user.id is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    # Use current_user instead of token
    session_id = str(uuid4())  # Convert UUID to string
    session.session_title = session.session_title or "Chat Session"

    await chat_service.create_session(current_user.id, session_id, session.session_title)
    return session_id


#send a message at the unique session id
@router.post("/send_message", status_code=status.HTTP_201_CREATED)
async def send_message(
    message: MessageContent,
    chat_service: ChatService = Depends(get_chat_service),
    current_user = Depends(get_current_user)
):
    #Verify user has access to this session
    if current_user.id is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")

    logger.info(f"Received message for session_id: {message.session_id} from user_id: {current_user.id}")
        
   # Create message stream
   #session id 0b8e8c6e-3a20-4056-b714-28dec35aeca2   d298b754-8b36-4b78-a484-ead9bf7a9181
   
    async def event_generator():
        try:
            async for chunk in chat_service.create_message_stream(
                session_id=message.session_id,
                user_id= current_user.id,  # Ensure this is the correct user_id
                content=message.content,
                role=message.role
            ):
                logger.debug(f"Yielding chunk: {chunk}")
                yield chunk
        except Exception as e:
            logger.error(f"Error during event generation: {str(e)}", exc_info=True)
            yield f"Error: {str(e)}"
    
    logger.info(f"Streaming response for session_id: {message.session_id}")
    return StreamingResponse(event_generator(), media_type="text/plain")

#get all chats for a specific session id
@router.get("/get_user_sessions", response_model=List[MessageUserInterface]) 
async def get_chat(
    chat_service: ChatService = Depends(get_chat_service),
    current_user = Depends(get_current_user)
) -> List[MessageUserInterface]:
    # Verify user has access to this session
    chat = await chat_service.get_user_sessions(current_user.id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    if str(chat[0].user_id) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to access this chat")
    return chat

@router.get("/get_chat_details", response_model=List[Messages.Messages])
async def get_chat_history(
    request: GetUserMessagesRequest,
    chat_service: ChatService = Depends(get_chat_service),
    current_user = Depends(get_current_user)
):
    if current_user.id is None:
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    return await chat_service.get_message_details(request.session_id)
    pass