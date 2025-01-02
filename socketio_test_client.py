# socketio_test_client.py
import socketio

# Initialize the SocketIO client
sio = socketio.Client()

@sio.event
def connect():
    print("Connected to the server")
    sio.emit('message', 'Hello Server!')

@sio.event
def response(data):
    print(f"Received from server: {data['message']}")
    sio.disconnect()

@sio.event
def disconnect():
    print("Disconnected from the server")

def main():
    try:
        sio.connect('http://localhost:3000')
        sio.wait()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()