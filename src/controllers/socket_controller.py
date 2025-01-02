import logging
from typing import Any
import asyncio

from services.chat_service import ChatService

# Import the instantiated ChatService from ServiceContainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set to INFO to see connection logs

class SocketController:
    def __init__(self, sio, chat_service: ChatService = None):
        if not chat_service:
            raise ValueError("ChatService instance is required")
        self.sio = sio
        self.chat_service = chat_service
        logger.info("SocketController initialized with ChatService")
        self.setup_events()

    def setup_events(self):
        @self.sio.event
        async def connect(sid, environ):
            """Handle client connection"""
            logger.info(f"Client connected: {sid}")
            await self.sio.emit('connect_response', {'status': 'connected'}, room=sid)

        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {sid}")

        @self.sio.event
        async def join(sid, data):
            """Handle room joining"""
            try:
                session_id = data.get('session_id')
                if not session_id:
                    raise ValueError("Session ID is required")
                
                await self.sio.enter_room(sid, session_id)
                logger.info(f"Client {sid} joined room {session_id}")
                
                # Send confirmation to client
                await self.sio.emit('joined', {
                    'status': 'success',
                    'message': f'Joined room {session_id}',
                    'session_id': session_id
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error in join event: {str(e)}")
                await self.sio.emit('error', {
                    'message': str(e)
                }, room=sid)

        @self.sio.event
        async def message(sid, data):
            """Handle incoming client messages and initiate AI streaming"""
            logger.info(f"Message from {sid}: {data}")
            try:
                if not self.chat_service:
                    logger.error("ChatService not available")
                    raise ValueError("ChatService not initialized")
                
                session_id = data.get('session_id')
                content = data.get('content')
                if not session_id or not content:
                    raise ValueError("Both session_id and content are required")
                
                logger.info(f"Processing message for session {session_id}")
                # Emit acknowledgment to client
                await self.sio.emit('response', {'message': 'Message received, processing...'}, to=sid)
                
                # Use the instance variable chat_service
                response = await self.chat_service.create_message(
                    session_id=session_id,
                    user_id=data.get('user_id', 'unknown'),
                    content=content,
                    role='human',
                    metadata=None
                )
                
            except Exception as e:
                logger.error(f"Error handling message event: {str(e)}")
                await self.sio.emit('error', {'message': str(e)}, room=sid)

        @self.sio.event
        async def stream_start(sid, data):
            """Handle start of AI message streaming"""
            session_id = data.get('session_id')
            if session_id:
                await self.sio.emit('stream_start', {'session_id': session_id}, room=session_id)

        @self.sio.event
        async def stream_token(sid, data):
            """Handle individual token streaming"""
            session_id = data.get('session_id')
            token = data.get('token')
            if session_id and token:
                await self.sio.emit('stream_token', {'token': token}, room=session_id)

        @self.sio.event
        async def stream_end(sid, data):
            """Handle end of AI message streaming"""
            session_id = data.get('session_id')
            if session_id:
                await self.sio.emit('stream_end', {'session_id': session_id}, room=session_id)

        @self.sio.event
        async def test_stream(sid, data):
            """Test endpoint for streaming"""
            try:
                session_id = data.get('session_id')
                if not session_id:
                    raise ValueError("Session ID is required")

                # Signal stream start
                await self.sio.emit('stream_start', {
                    'status': 'started',
                    'session_id': session_id
                }, room=session_id)

                # Test message
                test_message = "This is a test message that will be streamed word by word"
                for word in test_message.split():
                    await self.sio.emit('stream_token', {
                        'token': word + " ",
                        'session_id': session_id
                    }, room=session_id)
                    await asyncio.sleep(0.5)

                # Signal stream end
                await self.sio.emit('stream_end', {
                    'status': 'completed',
                    'session_id': session_id
                }, room=session_id)

            except Exception as e:
                logger.error(f"Error in test_stream: {str(e)}")
                await self.sio.emit('error', {
                    'message': str(e)
                }, room=sid)

    async def stream_ai_message(self, session_id: str, token: str, is_end: bool = False):
        """Helper method to stream AI messages"""
        if is_end:
            await self.sio.emit('stream_end', {'session_id': session_id}, room=session_id)
        else:
            await self.sio.emit('stream_token', {'token': token}, room=session_id)
