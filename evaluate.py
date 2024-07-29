# evaluate.py
from flask import Blueprint, jsonify
import tensorflow as tf
from keras._tf_keras.keras.preprocessing import ImageDataGenerator
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# Create a Blueprint for evaluation
evaluate_bp = Blueprint('evaluate', __name__)

# Path to the trained model and test data directory
model_path = 'saved_model/model.h5'
test_data_dir = 'uploads/test_data/'

@evaluate_bp.route('/evaluate', methods=['POST'])
def evaluate_model():
    global model_path, test_data_dir
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found. Please train a model first.'}), 404
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
        
        # Prepare the test data
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Predict on the test data
        predictions = model.predict(test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Return evaluation metrics
        return jsonify({
            'accuracy': float(model.evaluate(test_generator, verbose=0)[1]),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        })
        
    except Exception as e:
        return jsonify({'error': f"Error evaluating model: {e}"}), 500
