import os
import tensorflow as tf
import numpy as np
import cv2

MODEL_FILE = 'uploads/face_shape_model.h5'

def model_exists():
    return os.path.exists(MODEL_FILE)

def load_model():
    return tf.keras.models.load_model(MODEL_FILE)

def predict_image(image_path):
    if not model_exists():
        return {'error': 'Model file not found'}

    model = load_model()
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))
    img = np.array(img, dtype='float32') / 255.0
    img = np.expand_dims(img, axis=0)

    class_names = ['heart', 'oblong', 'oval', 'round', 'square']
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return {'class': predicted_class, 'confidence': f"{confidence:.2f}%"}
