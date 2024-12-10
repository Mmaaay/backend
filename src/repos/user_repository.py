from datetime import datetime
from typing import List, Optional
from bson import ObjectId
from pydantic import BaseModel
from models.dto import CreateUser, GetUser
from models.Users import User
from repos.base_repository import BaseRepository
from db.collections import Collections

class UserRepository(BaseRepository[User]):
    def __init__(self):
        super().__init__(Collections.users(), User)

    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Fetches a user by their email.
        
        :param email: The email of the user to fetch.
        :return: A User model instance if found, otherwise None.
        """
        
        user_data = await self.collection.find_one({"email": email})
        if user_data:
            user_data["_id"] = str(user_data["_id"])
            return self.model_class(**user_data)
        return None

    async def get_by_id(self, id: str) -> Optional[User]:
        """
        Fetches a user by their ID.
        
        :param id: The ID of the user to fetch.
        :return: A User model instance if found, otherwise None.
        """
        
        user_data = await self.collection.find_one({"_id": ObjectId(id)})
        if user_data:
            user_data["_id"] = str(user_data["_id"])
            return self.model_class(**user_data)
        return None

    async def get(self, limit: int = 1000, offset: int = 0) -> List[User]:
        """
        Fetches multiple users with optional limit and offset.

        :param limit: The maximum number of users to fetch.
        :param offset: The starting index for fetching.
        :return: A list of User model instances.
        """
        
        cursor = self.collection.find({}).skip(offset).limit(limit)
        users = []
        async for user_data in cursor:
            user_data["_id"] = str(user_data["_id"])
            users.append(self.model_class(**user_data))
        return users

    async def add(self, user: CreateUser) -> User:
        """
        Adds a new user to the collection.

        :param user: The CreateUser DTO to add.
        :return: The created User model instance.
        """
        user_data = user.model_dump() if isinstance(user, BaseModel) else user
        created_user = await super().create(self.model_class(**user_data))
        return created_user

    async def update(
        self, id: str, name: str, surname: str, role: str, email: str, password: str
    ) -> Optional[User]:
        """
        Updates a user's details.

        :param id: The ID of the user to update.
        :param name: Updated name of the user.
        :param surname: Updated surname of the user.
        :param role: Updated role of the user.
        :param email: Updated email of the user.
        :param password: Updated hashed password of the user.
        :return: The updated User model instance if successful, otherwise None.
        """
        update_data = {
            "name": name,
            "surname": surname,
            "role": role,
            "email": email,
            "password": password,
            "updated_at": datetime.now()
        }
        updated_user = await super().update(id, self.model_class(**update_data))
        return updated_user

    async def delete(self, id: str) -> bool:
        """
        Deletes a user by their ID.

        :param id: The ID of the user to delete.
        :return: True if deletion was successful, otherwise False.
        """
        return await super().delete(id)
