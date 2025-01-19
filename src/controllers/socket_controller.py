import logging
from typing import Any

logger = logging.getLogger(__name__)

class SocketController:
    def __init__(self, sio):
        self.sio = sio
        self.setup_events()

    def setup_events(self):
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")

        @self.sio.event
        async def disconnect(sid):
            logger.info(f"Client disconnected: {sid}")

        @self.sio.event
        async def join(sid, data):
            session_id = data.get('session_id')
            if session_id:
                self.sio.enter_room(sid, session_id)
                logger.info(f"Client {sid} joined room {session_id}")
                await self.sio.emit('joined', {'message': f'Joined room {session_id}'}, room=sid)
            else:
                logger.warning(f"Join event from {sid} missing session_id")

        @self.sio.event
        async def message(sid, data):
            logger.info(f"Message from {sid}: {data}")
            await self.sio.emit('response', {'message': 'Message received'}, to=sid)
