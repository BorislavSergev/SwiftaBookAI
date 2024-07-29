import os
import zipfile
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras import layers, models
import logging
import datetime
from PIL import Image
from flask_socketio import SocketIO

# Setup logging
logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads/'

# Placeholder variables for training status and logs
is_training = False
training_logs = []

# Function to append detailed log messages
def append_log(message):
    global training_logs
    log_entry = f"{datetime.datetime.now()}: {message}"
    training_logs.append(log_entry)
    logging.info(log_entry)  # Optionally log to console as well

# Function to extract zip file
def extract_zip(filepath):
    append_log(f"Starting zip extraction for file: {filepath}")
    extract_dir = os.path.join(UPLOAD_FOLDER, os.path.splitext(os.path.basename(filepath))[0])
    os.makedirs(extract_dir, exist_ok=True)  # Ensure directory exists
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    append_log(f"Zip extraction completed. Folder: {extract_dir}")
    return extract_dir

# Function to validate images
def validate_images(dataset_path):
    append_log(f"Starting image validation in folder: {dataset_path}")
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
                append_log(f"Invalid or corrupt image found and removed: {img_path}, Error: {e}")
                os.remove(img_path)
    append_log("Image validation completed.")

# Define custom metrics

def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.argmax(y_pred, axis=-1), tf.float32))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    
    tf.print("Precision - True Positives:", true_positives)
    tf.print("Precision - Predicted Positives:", predicted_positives)
    
    return precision

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))), tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(tf.argmax(y_true, axis=-1), tf.float32))
    
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    
    tf.print("Recall - True Positives:", true_positives)
    tf.print("Recall - Possible Positives:", possible_positives)
    
    return recall

def f1_score_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    tf.print("F1 Score - Precision:", precision)
    tf.print("F1 Score - Recall:", recall)
    
    return f1_score


# Define a custom callback to send logs to the client
class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, socketio):
        super().__init__()
        self.socketio = socketio

    def on_epoch_end(self, epoch, logs=None):
        log_entry = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            'precision_m': logs.get('precision_m'),
            'recall_m': logs.get('recall_m'),
            'f1_score_m': logs.get('f1_score_m')
        }
        # Emit the log entry to the client
        self.socketio.emit('log', log_entry)
        # Optionally print logs to console for debugging
        print(f"Epoch {epoch + 1}: {log_entry}")

# Function to start training
def start_training(filepath, socketio):
    global is_training
    is_training = True
    training_logs.clear()

    try:
        # Extract the uploaded zip file
        dataset_path = extract_zip(filepath)
        append_log("Dataset extraction successful.")

        # Validate images
        validate_images(dataset_path)
        append_log("Image validation successful.")

        # Prepare data generators
        append_log("Preparing data generators.")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2  # Split the data for validation
        )

        # Load training and validation data
        train_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='training'  # Set as training data
        )

        validation_generator = train_datagen.flow_from_directory(
            dataset_path,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            subset='validation'  # Set as validation data
        )

        # Build the model
        append_log("Building the model.")
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

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                 'accuracy',
                precision_m,  # Custom precision
                recall_m,     # Custom recall
                f1_score_m,   # Custom F1 score
                tf.keras.metrics.AUC(),  # Built-in AUC metric
            ]
        )

        # Define a custom callback to send logs to the client
        training_callback = TrainingCallback(socketio)

        append_log("Starting model training.")
        # Train the model
        model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            callbacks=[training_callback]
        )

        # Save the model
        model.save('saved_model/model.h5')

        append_log("Training completed successfully.")
        is_training = False
        return 'Training completed', 200

    except Exception as e:
        append_log(f"Error during training: {e}")
        is_training = False
        return f"Error during training: {e}", 500

# Function to get training status
def get_training_status():
    return {'is_training': is_training}, 200

# Function to get logs
def get_logs():
    return {'logs': training_logs}, 200

# Function to clear logs
def clear_logs():
    global training_logs
    training_logs.clear()
    return 'Logs cleared', 200

# Function to get machine stats
def get_machine_stats():
    import platform
    import psutil
    import datetime

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

# Dummy implementation for image prediction
def predict_image(file):
    return {"prediction": "Not implemented"}, 200
