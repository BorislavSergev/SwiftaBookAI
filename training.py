import os
import zipfile
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
import psutil
import platform
import logging
import datetime
from PIL import Image
from flask import jsonify

# Define log file path
LOG_FILE_PATH = 'logs/training_logs.txt'

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Initialize logger
logging.basicConfig(level=logging.INFO)

# Placeholder variables for training status and logs
is_training = False
training_logs = []

# Function to append messages to logs
def log_message(message, level='INFO'):
    # Append message to in-memory log list
    training_logs.append(f"{datetime.datetime.now()} - {level} - {message}")
    
    # Append message to log file
    with open(LOG_FILE_PATH, 'a') as log_file:
        log_file.write(f"{datetime.datetime.now()} - {level} - {message}\n")
    
    # Print the message with color
    if level == 'INFO':
        print(f"\033[92m{message}\033[0m")  # Green for INFO
    elif level == 'ERROR':
        print(f"\033[91m{message}\033[0m")  # Red for ERROR

# Function to extract zip file
def extract_zip(filepath):
    extract_dir = os.path.join('uploads', os.path.splitext(os.path.basename(filepath))[0])
    os.makedirs(extract_dir, exist_ok=True)  # Ensure directory exists
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        log_message("Starting zip extraction", 'INFO')
        zip_ref.extractall(extract_dir)
        log_message(f"Zip extraction completed, folder={extract_dir}", 'INFO')
    return extract_dir

# Function to validate images
def validate_images(dataset_path):
    log_message("Starting image validation", 'INFO')
    for root, _, files in os.walk(dataset_path):
        for file in files:
            try:
                img_path = os.path.join(root, file)
                # Use PIL to open the image
                with Image.open(img_path) as img:
                    img.verify()  # Will raise an exception if the image is corrupt
                    img = Image.open(img_path)
                    img = img.resize((150, 150))
            except (IOError, SyntaxError) as e:
                log_message(f"Invalid or corrupt image found and removed: {img_path}, Error: {e}", 'ERROR')
                os.remove(img_path)
    log_message("Image validation completed", 'INFO')

# Function to start training
def start_training(filepath, socketio):
    global is_training
    is_training = True
    training_logs.clear()
    
    try:
        # Extract the uploaded zip file
        dataset_path = extract_zip(filepath)

        # Validate images
        validate_images(dataset_path)

        # Prepare data generators
        log_message("Preparing data generators", 'INFO')
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
        log_message("Building the model", 'INFO')
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
            layers.Dense(train_generator.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Define a custom callback to send logs to the client
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                log_entry = (f"Epoch {epoch + 1}: Loss={logs.get('loss')}, Accuracy={logs.get('accuracy')}, "
                             f"Val Loss={logs.get('val_loss')}, Val Accuracy={logs.get('val_accuracy')}")
                log_message(log_entry, 'INFO')
                socketio.emit('log', {'message': log_entry})

        # Train the model
        log_message("Starting training", 'INFO')
        model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=[TrainingCallback()]
        )

        log_message("Training completed", 'INFO')
        is_training = False
        return 'Training completed', 200

    except Exception as e:
        log_message(f"Error during training: {e}", 'ERROR')
        is_training = False
        return f"Error during training: {e}", 500

# Function to get training status
def get_training_status():
    return {'is_training': is_training}, 200

# Function to get logs
def get_logs():
    with open(LOG_FILE_PATH, 'r') as log_file:
        logs = log_file.readlines()
    return {'logs': logs}, 200

# Function to clear logs
def clear_logs():
    # Clear in-memory logs
    training_logs.clear()
    
    # Clear log file
    with open(LOG_FILE_PATH, 'w') as log_file:
        log_file.write('')
    
    return 'Logs cleared', 200

# Function to get machine stats
def get_machine_stats():
    cpu_info = platform.processor()
    system_info = platform.system()
    release_info = platform.release()
    ram_info = f"{round(psutil.virtual_memory().total / (1024.0 **3))} GB"
    uptime_seconds = psutil.boot_time()
    uptime = str(datetime.timedelta(seconds=int(uptime_seconds)))
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
    return stats, 200

# Dummy implementation for model evaluation
def evaluate_model():
    return "Model evaluation not implemented", 200

# Dummy implementation for image prediction
def predict_image(file):
    return {"prediction": "Not implemented"}, 200
