import os
import zipfile
import tensorflow as tf
from flask import jsonify
from flask_socketio import SocketIO
import psutil
import time
import cv2
import numpy as np
from utils import check_directories, log_directory_structure
from sklearn.metrics import classification_report

UPLOAD_FOLDER = 'uploads'
LOG_FILE = 'training_log.txt'
MODEL_FILE = os.path.join(UPLOAD_FOLDER, 'face_shape_model.h5')

training_state = {'is_training': False}

def start_training(filepath, socketio):
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_FOLDER)
        print("Zip file extracted successfully.")
    except zipfile.BadZipFile:
        print("Invalid zip file.")
        return 'Invalid zip file', 400

    os.remove(filepath)
    print(f"File {filepath} removed after extraction.")

    log_directory_structure(UPLOAD_FOLDER)

    if not check_directories(UPLOAD_FOLDER):
        print("Training or validation directory not found after extraction.")
        log_directory_structure(UPLOAD_FOLDER)
        return 'Training or validation directory not found after extraction', 400

    if not training_state['is_training']:
        training_state['is_training'] = True
        socketio.start_background_task(target=train_model, folder=UPLOAD_FOLDER, socketio=socketio)
        return 'Training started', 200
    else:
        return 'Training already in progress', 400

def get_training_status():
    return jsonify(training_state)

def get_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as file:
            logs = file.read()
        return logs
    else:
        return jsonify({'error': 'Log file not found'}), 404

def clear_logs():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return 'Logs cleared', 200

def get_machine_stats():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory().percent
    uptime_seconds = time.time() - psutil.boot_time()

    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s"

    cores = psutil.cpu_count(logical=False)
    
    stats = {
        'cpu_usage': cpu_usage,
        'memory_info': memory_info,
        'uptime': uptime_str,
        'cores': cores
    }
    return jsonify(stats)

def evaluate_model():
    model_path = MODEL_FILE
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404

    model = tf.keras.models.load_model(model_path)
    val_data, val_labels = load_data(UPLOAD_FOLDER, subset='validation')

    predictions = model.predict(val_data)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(val_labels, axis=1)

    report = classification_report(true_labels, predicted_labels, target_names=['heart', 'oblong', 'oval', 'round', 'square'], output_dict=True)

    accuracy = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']

    accuracy_file = 'model_accuracy.txt'
    with open(accuracy_file, 'w') as file:
        file.write(f"Validation Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Precision: {precision * 100:.2f}%\n")
        file.write(f"Recall: {recall * 100:.2f}%\n")
        file.write(f"F1 Score: {f1_score * 100:.2f}%\n")

    return jsonify({
        'accuracy': f"{accuracy * 100:.2f}%",
        'precision': f"{precision * 100:.2f}%",
        'recall': f"{recall * 100:.2f}%",
        'f1_score': f"{f1_score * 100:.2f}%"
    })
def load_data(folder, subset='training'):
    data = []
    labels = []
    class_names = ['heart', 'oblong', 'oval', 'round', 'square']
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (150, 150))
                data.append(img)
                labels.append(class_names.index(class_name))
    data = np.array(data, dtype='float32') / 255.0
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names)) 
    return data, labels

def train_model(folder, socketio):
    try:
        model = create_model()

        train_data, train_labels = load_data(folder)

        # Split the data into training and validation sets
        validation_split = 0.2
        num_validation_samples = int(validation_split * len(train_data))
        val_data = train_data[:num_validation_samples]
        val_labels = train_labels[:num_validation_samples]
        train_data = train_data[num_validation_samples:]
        train_labels = train_labels[num_validation_samples:]

        class CustomCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                with open(LOG_FILE, 'a') as log_file:
                    log_file.write(f"Epoch {epoch + 1}: Loss={logs['loss']}, Accuracy={logs['accuracy']}, "
                                   f"Val Loss={logs['val_loss']}, Val Accuracy={logs['val_accuracy']}\n")
                socketio.emit('log', {
                    'epoch': epoch,
                    'loss': logs['loss'],
                    'accuracy': logs['accuracy'],
                    'val_loss': logs['val_loss'],
                    'val_accuracy': logs['val_accuracy']
                })

        model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=10,
            callbacks=[CustomCallback()]
        )

        model.save(MODEL_FILE)

    except Exception as e:
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"Error during training: {str(e)}\n")

    finally:
        training_state['is_training'] = False

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 face shape classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
