from typing import Generic, TypeVar, Optional, Type
from pydantic import BaseModel
from bson import ObjectId
from pymongo.collection import Collection
from models.Base import BaseDBModel

T = TypeVar('T', bound=BaseDBModel)

class BaseRepository(Generic[T]):
    def __init__(self, collection: Collection, model_class: Type[T]):
        self.collection = collection
        self.model_class = model_class
    
    async def find_by_id(self, id: str) -> Optional[T]:
        try:
            result = await self.collection.find_one({"_id": ObjectId(id)})
            if result:
                result["_id"] = str(result["_id"])
                return self.model_class(**result)
            return None
        except Exception:
            return None
    
    async def create(self, model: T) -> T:
        model_dict = model.model_dump(exclude={'id'}) if isinstance(model, BaseModel) else model
        result = await self.collection.insert_one(model_dict)
        model_dict['_id'] = str(result.inserted_id)
        return self.model_class(**model_dict)
    
    async def update(self, id: str, model: T) -> Optional[T]:
        try:
            update_data = model.model_dump(exclude_unset=True, exclude={'id'}) if isinstance(model, BaseModel) else model
            result = await self.collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": update_data}
            )
            if result.modified_count > 0:
                return await self.find_by_id(id)
            return None
        except Exception:
            return None
    
    async def delete(self, id: str) -> bool:
        try:
            result = await self.collection.delete_one({"_id": ObjectId(id)})
            return result.deleted_count > 0
        except Exception:
            return False
