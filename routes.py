from flask import render_template, request, jsonify
from training import start_training, get_training_status, clear_logs, evaluate_model, get_logs, get_machine_stats
import os

UPLOAD_FOLDER = 'uploads'

def setup_routes(app, socketio):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/train', methods=['POST'])
    def train():
        if 'file' not in request.files:
            return 'No file part', 400

        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        try:
            file.save(filepath)
        except Exception as e:
            return str(e), 500

        response = start_training(filepath, socketio)
        return response

    @app.route('/training-status', methods=['GET'])
    def training_status():
        return get_training_status()

    @app.route('/logs', methods=['GET'])
    def logs():
        return get_logs()

    @app.route('/clear-logs', methods=['GET'])
    def clear_logs_route():
        return clear_logs()

    @app.route('/machine-stats', methods=['GET'])
    def machine_stats():
        return get_machine_stats()

    @app.route('/accuracy')
    def accuracy():
        return render_template('accuracy.html')

    @app.route('/training')
    def training():
        return render_template('training.html')

    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        return evaluate_model()
