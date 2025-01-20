from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from utils.token import create_token, decode_token
from models import dto

from services.user_service import UserService, get_user_service
from utils import formating , dependencies 
from utils.formating import MongoIDConverter
from models import Users
from utils.bcrypt_hashing import HashLib
from constants import  COOKIES_KEY_NAME

router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
)

@router.post("/signup", response_model=str,status_code=status.HTTP_201_CREATED)
async def signup(user: dto.CreateUser  ,res:Response , user_service: UserService = Depends(get_user_service)) -> Users.User:
    email = user.email
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email")
    email = formating.format_string(user.email)
    
    if not user.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid password")
    password = HashLib.hash(user.password)
    
    existing_user = await user_service.get_user_by_email(email)
    
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists")
    
    user = {
        "name": user.name,
        "email": email,
        "role": Users.User.Role.USER,
        "password": password,
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    returned_user = await user_service.create_user(
        user
    )
    print(returned_user)
    
    token = create_token({ "id" : returned_user.id , "name": returned_user.name, "email": returned_user.email , "role": returned_user.role})    
    return token

@router.post("/login", response_model=str , status_code=status.HTTP_200_OK)
async def login(dto: dto.LoginUser , res: Response, user_service: UserService = Depends(get_user_service)) -> Users.User:

    email = formating.format_string(dto.email)
    user = await user_service.get_user_by_email(email)
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    if not HashLib.validate(dto.password, user.password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid password")
    
    token = create_token({ "id" : user.id , "name": user.name, "email": user.email , "role": user.role})    
    

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"access_token": token, "token_type": "bearer"}
    )
    #string@string.string



@router.post("/password/update", status_code=204)
async def update_password(
    dto: dto.UpdateUserPass,
    current_user = Depends(dependencies.get_current_user),
    user_service: UserService = Depends(get_user_service)
):
    user = await user_service.get_user_by_email(dto.email)
    
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not valid")

    if not HashLib.validate(dto.old_password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    
    user_id = MongoIDConverter.ensure_object_id(user.id)
    await user_service.update_password(user_id, dto.new_password)
    return None


