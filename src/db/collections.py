from typing import Optional
from pymongo.collection import Collection
from db.mongo_client import MongoDBClient

class Collections:
    _users: Optional[Collection] = None
    _chat_sessions: Optional[Collection] = None
    _messages: Optional[Collection] = None
    _tajweed_sessions: Optional[Collection] = None
    
    @classmethod
    def users(cls) -> Collection:
        if cls._users is None:
            cls._users = MongoDBClient.get_db().users
        return cls._users
    
    @classmethod
    def chat_sessions(cls) -> Collection:
        if cls._chat_sessions is None:
            cls._chat_sessions = MongoDBClient.get_db().chat_sessions
        return cls._chat_sessions
    
    @classmethod
    def messages(cls) -> Collection:
        if cls._messages is None:
            cls._messages = MongoDBClient.get_db().messages
        return cls._messages
    
    @classmethod
    def tajweed_sessions(cls) -> Collection:
        if cls._tajweed_sessions is None:
            cls._tajweed_sessions = MongoDBClient.get_db().tajweed_sessions
        return cls._tajweed_sessions


