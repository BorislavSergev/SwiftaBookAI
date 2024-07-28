from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('path_to_your_model.h5')  # Replace with your model's path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Use cv2 to load and process the image
            img = cv2.imread(filepath)
            img = cv2.resize(img, (224, 224))  # Adjust size to match your model input
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            face_shapes = ['Oval', 'Round', 'Square', 'Heart', 'Diamond']  # Adjust based on your model's classes
            predicted_shape = face_shapes[predicted_class]

            os.remove(filepath)  # Clean up the uploaded file
            return jsonify({'prediction': predicted_shape})

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
