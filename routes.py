from flask import render_template, request, jsonify
from training import start_training, get_training_status, clear_logs, evaluate_model, get_logs, get_machine_stats, predict_image
import os

UPLOAD_FOLDER = 'uploads/'

def setup_routes(app, socketio):
    @app.route('/machine-stats', methods=['GET'])
    def machine_stats():
        stats, status_code = get_machine_stats()
        return jsonify(stats), status_code

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
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
        except Exception as e:
            return str(e), 500

        response = start_training(filepath, socketio)
        return jsonify(response)

    @app.route('/training-status', methods=['GET'])
    def training_status():
        return jsonify(get_training_status())

    @app.route('/logs', methods=['GET'])
    def logs():
        return jsonify(get_logs())

    @app.route('/clear-logs', methods=['GET'])
    def clear_logs_route():
        return jsonify(clear_logs())

    @app.route('/machine-stats', methods=['GET'])
    def machine_stats():
        return jsonify(get_machine_stats())

    @app.route('/accuracy')
    def accuracy():
        return render_template('accuracy.html')

    @app.route('/training')
    def training():
        return render_template('training.html')

    @app.route('/evaluate', methods=['POST'])
    def evaluate():
        return jsonify(evaluate_model())

    @app.route('/predict', methods=['GET'])
    def predict_page():
        return render_template('predict.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        response = predict_image(file)
        return jsonify(response)
