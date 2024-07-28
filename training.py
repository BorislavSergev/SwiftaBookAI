
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import psutil
import platform
import logging
from flask import Flask, request, jsonify
from flask_socketio import SocketIO

logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads/'

# Placeholder variables for training status and logs
is_training = False
training_logs = []

app = Flask(__name__)
socketio = SocketIO(app)

# Function to extract zip file
def extract_zip(filepath):
    extract_dir = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(filepath))[0])
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir

# Function to start training
def start_training(filepath, socketio):
    global is_training
    is_training = True
    training_logs.clear()
    
    try:
        # Extract the uploaded zip file
        dataset_path = extract_zip(filepath)
        logging.info(f"Dataset extracted to {dataset_path}")

        # Prepare data generators
        train_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(256, 256),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        # Build the model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(train_generator.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define a custom callback to send logs to the client
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                log_entry = f"Epoch {epoch + 1}: Loss={logs.get('loss')}, Accuracy={logs.get('accuracy')}, Val Loss={logs.get('val_loss')}, Val Accuracy={logs.get('val_accuracy')}"
                training_logs.append(log_entry)
                socketio.emit('log', {
                    'epoch': epoch,
                    'loss': logs.get('loss'),
                    'accuracy': logs.get('accuracy'),
                    'val_loss': logs.get('val_loss'),
                    'val_accuracy': logs.get('val_accuracy')
                })

        # Train the model
        model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=[TrainingCallback()]
        )

        logging.info("Training completed")
        is_training = False
        return 'Training started', 200

    except Exception as e:
        logging.error(f"Error during training: {e}")
        is_training = False
        return f"Error during training: {e}", 500

# Function to get training status
@app.route('/training-status', methods=['GET'])
def get_training_status():
    return jsonify({"is_training": is_training}), 200

# Function to get logs
@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': training_logs}), 200

# Function to clear logs
@app.route('/clear-logs', methods=['POST'])
def clear_logs():
    training_logs.clear()
    return 'Logs cleared', 200

# Function to get machine stats
@app.route('/machine-stats', methods=['GET'])
def get_machine_stats():
    cpu_info = platform.processor()
    system_info = platform.system()
    release_info = platform.release()
    ram_info = f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    uptime_seconds = psutil.boot_time()
    uptime = convert_seconds_to_time_format(uptime_seconds)
    cores_info = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory().percent
    
    stats = {
        "cpu_usage": cpu_usage,
        "memory_info": memory_info,
        "uptime": uptime,
        "cores": cores_info,
        "CPU": cpu_info,
        "System": system_info,
        "Release": release_info,
        "RAM": ram_info,
        "GPU": "NVIDIA"  # Placeholder, replace with actual GPU info if available
    }
    return jsonify(stats), 200

def convert_seconds_to_time_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    mo, d = divmod(d, 30)
    return f"{mo}m {d}d {h}h {m}min {s}s"

# Dummy implementation for model evaluation
@app.route('/evaluate-model', methods=['POST'])
def evaluate_model():
    return "Model evaluation not implemented", 200

# Dummy implementation for image prediction
@app.route('/predict-image', methods=['POST'])
def predict_image():
    return {"prediction": "Not implemented"}, 200

if __name__ == '__main__':
    socketio.run(app, debug=True)