import numpy as np
import tensorflow as tf
import os
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score

# Define custom metrics
def precision_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(tf.argmax(y_pred, axis=-1), tf.float32))
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_m(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))), tf.float32))
    possible_positives = tf.reduce_sum(tf.cast(tf.argmax(y_true, axis=-1), tf.float32))
    
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1_score_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    
    f1_score = 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    return f1_score

def load_test_data(test_data_dir):
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        x_test, y_test = next(test_generator)  # Load one batch for simplicity
        for i in range(1, len(test_generator)):
            x_batch, y_batch = next(test_generator)
            x_test = np.vstack((x_test, x_batch))
            y_test = np.vstack((y_test, y_batch))
        return x_test, y_test
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None

def calculate_metrics(model, x_test, y_test):
    # Evaluate the model
    results = model.evaluate(x_test, y_test, verbose=0)
    metrics = dict(zip(model.metrics_names, results))
    return metrics

def evaluate_model():
    model = tf.keras.models.load_model('saved_model/model.h5', custom_objects={
        'precision_m': precision_m,
        'recall_m': recall_m,
        'f1_score_m': f1_score_m
    })
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            precision_m,
            recall_m,
            f1_score_m,
            tf.keras.metrics.AUC(name='auc_roc'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR')
        ]
    )

    x_test, y_test = load_test_data('uploads/testData')
    
    if x_test is None or y_test is None:
        print("Failed to load test data.")
        raise ValueError("Test data could not be loaded.")
    
    print(f"Loaded test data: x_test shape={x_test.shape}, y_test shape={y_test.shape}")
    metrics = calculate_metrics(model, x_test, y_test)
    print(f"Calculated metrics: {metrics}")
    
    return metrics

from PIL import Image
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img

# Load your model
def load_model():
    model = tf.keras.models.load_model('saved_model/model.h5', compile=False)
    return model

model = load_model()

def predict_image(filepath):
    # Load the image and preprocess it
    img = load_img(filepath, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Map the prediction to class labels
    class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']  # Adjust according to your model's classes
    predicted_label = class_labels[predicted_class]

    return {"prediction": predicted_label}, 200
