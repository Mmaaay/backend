import socketio
import asyncio

# Create a Socket.IO asynchronous client
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print("Connected to server")
    # Join a room after connecting
    await sio.emit('join', {'session_id': 'test123'})
    # Optionally, send a message to trigger AI streaming
    await sio.emit('message', {
        'session_id': 'test123',
    'user_id': 'user_1',
        'content': 'What is the meaning of Surah Al-Fatiha?',
        'role': 'human',
        'metadata': {
            "session_id": "test123"
            
        }
    })

@sio.event
async def connect_error(data):
    print("Connection failed")

@sio.event
async def disconnect():
    print("Disconnected from server")

@sio.event
async def connect_response(data):
    print(f"Connect Response: {data}")

@sio.event
async def joined(data):
    print(f"Joined Room: {data}")

@sio.event
async def stream_start(data):
    print(f"Stream Started: {data}")

@sio.event
async def stream_token(data):
    print(f"Received Token: {data['token']}")

@sio.event
async def stream_end(data):
    print(f"Stream Ended: {data}")

@sio.event
async def error(data):
    print(f"Error: {data['message']}")

@sio.event
async def response(data):
    token = data.get('response', '')
    is_end = data.get('is_end', False)
    
    if not hasattr(sio, 'complete_response'):
        sio.complete_response = ""
    
    sio.complete_response += token
    
    if is_end:
        print(f"Received Complete Response: {sio.complete_response}")
        sio.complete_response = ""  # Reset for next response
    else:
        if token:  # Only print if token is not empty
            print(f"Received Token: {token}")

async def main():
    try:
        # Ensure the client connects to the correct URL with the appropriate path
        await sio.connect('http://localhost:3000')  # Update if different
        await sio.wait()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())