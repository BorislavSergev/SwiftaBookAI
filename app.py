from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from routes import setup_routes

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

setup_routes(app, socketio)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, use_reloader=False)
