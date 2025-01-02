import asyncio
import socketio

sio = socketio.AsyncClient()

async def main():
    try:
        await sio.connect('http://localhost:3000')
        print("Connected to server")

        @sio.event
        async def ai_response(data):
            print(f"Received AI response chunk: {data['chunk']}")

        @sio.event
        async def joined(data):
            print(f"Joined room: {data['message']}")

        # Join a room with a specific session_id
        session_id = 'test_session'
        await sio.emit('join', {'session_id': session_id})
        print(f"Sent join request for session_id: {session_id}")

        # Send a message to the server
        await sio.emit('message', {'session_id': session_id, 'content': 'Hello AI'})
        print("Sent message: 'Hello AI'")

        # Keep the client running to listen for incoming messages
        await sio.wait()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())