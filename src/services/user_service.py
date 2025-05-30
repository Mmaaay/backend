from typing import List, Optional
from random import randint
from models.dto import CreateUser, GetUser
from models.Users import User
from repos.user_repository import UserRepository
from utils.bcrypt_hashing import HashLib
from fastapi import Depends

class UserService:
    def __init__(self):
        self.user_repo = UserRepository()
    
    async def get_users(self, limit: int = 1000, offset: int = 0) -> List[User]:
        
        return await self.user_repo.get(limit=limit, offset=offset)

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        return await self.user_repo.get_by_id(user_id)
    
    async def get_user_by_email(self, email: str) -> Optional[GetUser]:
        return await self.user_repo.get_by_email(email.lower().strip())

    async def create_user(self, user: CreateUser) -> User:
        return await self.user_repo.add(user)
    
    async def update_password(self, user_id: str, new_password: str) -> None:
        user = await self.get_user_by_id(user_id)
        if not user:
            return
        new_pass_hash = HashLib.hash(new_password)
        await self.user_repo.update(
            id=user_id,
            name=user.name,
            role=user.role,
            email=user.email,
            password=new_pass_hash
        )
    
    async def reset_password(self, user_id: str) -> Optional[str]:
        user = await self.get_user_by_id(user_id)
        if not user:
            return None
        new_password = str(randint(1000, 9999))
        password_hash = HashLib.hash(new_password)
        await self.user_repo.update(
            id=user_id,
            name=user.name,
            surname=user.surname,
            role=user.role,
            email=user.email,
            password=password_hash
        )
        return new_password

    async def delete_user(self, user_id: str) -> None:
        await self.user_repo.delete(user_id)

def get_user_service() -> UserService:
    return UserService()


