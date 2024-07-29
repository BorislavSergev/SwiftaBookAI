from flask import jsonify
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator  # Correct import path
import numpy as np
import os

# Function to evaluate the model
def evaluate_model():
    try:
        # Load the model
        model = tf.keras.models.load_model('model.h5')

        # Prepare the data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            'test_data/',  # Make sure you have a test dataset
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )

        # Evaluate the model
        evaluation = model.evaluate(test_generator, verbose=1)
        metrics = model.metrics_names
        results = dict(zip(metrics, evaluation))

        # Convert results to a more readable format
        accuracy = results.get('accuracy', None)
        precision = results.get('precision', None)
        recall = results.get('recall', None)
        f1_score = results.get('f1_score', None)

        # Return results in a JSON format
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }, 200

    except Exception as e:
        return {'error': str(e)}, 500
