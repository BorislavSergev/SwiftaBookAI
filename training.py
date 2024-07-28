import datetime
import os
import zipfile
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
import psutil
import platform
import logging
import cv2
from flask import jsonify
import time

logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads/'

# Placeholder variables for training status and logs
is_training = False
training_logs = []

# Function to extract zip file
def extract_zip(filepath):
    extract_dir = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(filepath))[0])
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir

# Function to validate images
def validate_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(root, file)
                try:
                    img = cv2.imread(image_path)
                    if img is None or img.size == 0:
                        logging.error(f"Invalid image found and removed: {image_path}")
                        os.remove(image_path)  # Remove invalid images
                except Exception as e:
                    logging.error(f"Error loading image {image_path}: {e}")
                    os.remove(image_path)  # Remove problematic images

# Function to start training
def start_training(filepath, socketio):
    global is_training
    is_training = True
    training_logs.clear()
    
    try:
        # Extract the uploaded zip file
        dataset_path = extract_zip(filepath)
        logging.info(f"Dataset extracted to {dataset_path}")

        # Validate images in the dataset
        validate_images(dataset_path)

        # Prepare data generators
        train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )

        # Build the model
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(5, activation='softmax')
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
        return jsonify({"error": f"Error during training: {e}"}), 500

# Function to get training status
def get_training_status():
    return {"is_training": is_training}, 200

# Function to get logs
def get_logs():
    return '\n'.join(training_logs), 200

# Function to clear logs
def clear_logs():
    training_logs.clear()
    return 'Logs cleared', 200

# Function to get machine stats

def get_machine_stats():
    cpu_info = platform.processor()
    system_info = platform.system()
    release_info = platform.release()
    ram_info = f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    uptime_seconds = int(psutil.boot_time())
    uptime_delta = datetime.datetime.now() - datetime.datetime.fromtimestamp(uptime_seconds)
    uptime_info = f"{uptime_delta.days // 30}m {uptime_delta.days % 30}d {uptime_delta.seconds // 3600}h {(uptime_delta.seconds % 3600) // 60}min {uptime_delta.seconds % 60}s"
    cores_info = psutil.cpu_count(logical=True)
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory().percent
    
    stats = {
        "cpu_usage": cpu_usage,
        "memory_info": memory_info,
        "uptime": uptime_info,
        "cores": cores_info,
        "CPU": cpu_info,
        "System": system_info,
        "Release": release_info,
        "RAM": ram_info,
        "GPU": "NVIDIA"  # Placeholder, replace with actual GPU info if available
    }
    return stats, 200

# Dummy implementation for model evaluation
def evaluate_model():
    return "Model evaluation not implemented", 200

# Dummy implementation for image prediction
def predict_image(file):
    return {"prediction": "Not implemented"}, 200
