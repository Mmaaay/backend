from bson.objectid import ObjectId
from typing import Optional, Union

def format_string(email: str) -> str:
    return email.strip() \
                .replace(" ", "") \
                .replace("\n","") \
                .replace("\r","") \
                .lower()
                
class MongoIDConverter:
    @staticmethod
    def to_object_id(id: Union[str, ObjectId, None]) -> Optional[ObjectId]:
        """
        Convert a string ID or ObjectId to ObjectId.
        
        Args:
            id: String ID, ObjectId, or None
            
        Returns:
            ObjectId if conversion successful, None if input is None or invalid
        """
        try:
            if id is None:
                return None
            if isinstance(id, ObjectId):
                return id
            return ObjectId(str(id))
        except Exception as e:
            print(f"Error converting to ObjectId: {e}")
            return None

    @staticmethod
    def to_string(id: Union[ObjectId, str, None]) -> Optional[str]:
        """
        Convert an ObjectId or string ID to string.
        
        Args:
            id: ObjectId, string ID, or None
            
        Returns:
            String representation of ID if conversion successful, None if input is None or invalid
        """
        try:
            if id is None:
                return None
            if isinstance(id, str):
                return id
            return str(id)
        except Exception as e:
            print(f"Error converting to string: {e}")
            return None

    @staticmethod
    def ensure_object_id(id: Union[str, ObjectId, None]) -> Optional[ObjectId]:
        """
        Ensure the ID is an ObjectId. Convert if necessary.
        Useful for database queries.
        """
        return MongoIDConverter.to_object_id(id)

    @staticmethod
    def ensure_string(id: Union[ObjectId, str, None]) -> Optional[str]:
        """
        Ensure the ID is a string. Convert if necessary.
        Useful for API responses.
        """
        return MongoIDConverter.to_string(id)