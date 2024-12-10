from fastapi import APIRouter , status , Request , HTTPException, Depends
from services.chat_service import ChatService
from models.dto import MessageResponse
from services import jwt_service
from constants import COOKIES_KEY_NAME
from models import Messages
from uuid import uuid4
from typing import List
from datetime import datetime


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

def get_chat_service():
    return ChatService()

@router.post("/create_chat_session" , response_model=Messages.ChatSession , status_code=status.HTTP_201_CREATED)
async def create_chat(   req:Request ,  chat_service : ChatService = Depends(get_chat_service)) -> Messages.ChatSession:
    token = req.cookies.get(COOKIES_KEY_NAME)
    data = jwt_service.decode(token)
    if data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    unique_chat_id = str(uuid4())
    chat_title = data.user_id + unique_chat_id
    return await chat_service.create_session( data.user_id,unique_chat_id  , chat_title)


# @router.post("/send_message" , response_model=Messages.Messages , status_code=status.HTTP_201_CREATED)
# async def send_message(message: Messages.MessageContent , req:Request, chat_service : ChatService = Depends(get_chat_service)) -> Messages.Messages:
#     token = req.cookies.get(COOKIES_KEY_NAME)
#     random_id = str(uuid4())
#     data = jwt_service.decode(token)
#     if data is None:
#         raise HTTPException(status_code=401, detail="Unauthorized")
    
#     session_id = f"${random_id + "-" + data.user_id}"
 
#     return await chat_service.create_message( user_id=data.user_id , session_id=session_id , content=message , role="user" , metadata={
#         'user_id':data.user_id,
#         'session_id':session_id
#     })
    
    
@router.post("/send_message/{session_id}" , response_model=Messages.Messages , status_code=status.HTTP_201_CREATED)
async def send_message(session_id:str , message: Messages.MessageContent , req:Request, chat_service : ChatService = Depends(get_chat_service)) -> MessageResponse:
    token = req.cookies.get(COOKIES_KEY_NAME)
    data = jwt_service.decode(token)
    if data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    session_id: str = session_id
    
    return await chat_service.create_message( user_id=data.user_id , session_id=session_id , content=message , role="user" , metadata={
        'user_id':data.user_id,
        'session_id':session_id,
        'time_stamp': datetime.now().isoformat()
    })
    

@router.get("/get_messages/{session_id}", response_model=List[Messages.Messages], status_code=status.HTTP_200_OK)
async def get_messages(session_id: str, req: Request, chat_service: ChatService = Depends(get_chat_service)) -> List[Messages.Messages]:
    token = req.cookies.get(COOKIES_KEY_NAME)
    data = jwt_service.decode(token)
    if data is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    messages = await chat_service.get_chat_history(session_id)
    return messages