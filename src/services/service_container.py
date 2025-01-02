from typing import Optional
import logging
from services.chat_service import ChatService
from controllers.socket_controller import SocketController
from db.mongo_client import MongoDBClient  # Ensure this import exists and is correct

logger = logging.getLogger(__name__)

class ServiceContainer:
    _instance = None
    chat_service: Optional[ChatService] = None
    socket_controller: Optional[SocketController] = None
    db = None  # Added to access the database

    @classmethod
    async def initialize(cls, sio):
        if not cls._instance:
            cls._instance = cls()
            logger.info("Initializing services...")
            
            # Initialize the database
            cls.db = MongoDBClient.get_db()  # Updated to use MongoDBClient
            
            # First create ChatService with db
            cls.chat_service = ChatService(db=cls.db)
            logger.info("ChatService initialized")
            
            # Then create SocketController with ChatService
            cls.socket_controller = SocketController(sio, cls.chat_service)
            logger.info("SocketController initialized with ChatService")
            
            # Finally update ChatService with socket_controller reference
            cls.chat_service.socket_controller = cls.socket_controller
            logger.info("Services initialization complete")
            
        return cls._instance

    @classmethod
    def get_chat_service(cls):
        if not cls.chat_service:
            raise ValueError("ChatService not initialized. Call initialize() first.")
        return cls.chat_service

    @classmethod
    def get_socket_controller(cls):
        if not cls.socket_controller:
            raise ValueError("SocketController not initialized. Call initialize() first.")
        return cls.socket_controller
