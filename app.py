from flask import Flask
from flask_socketio import SocketIO
from routes import setup_routes

app = Flask(__name__, static_folder='static')
socketio = SocketIO(app)

setup_routes(app, socketio)

if __name__ == '__main__':
    socketio.run(app, debug=True)
