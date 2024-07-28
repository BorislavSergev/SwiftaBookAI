from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('path_to_your_model.h5')  # Replace with your model's path

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

            img = image.load_img(filepath, target_size=(224, 224))  # Adjust target_size to match your model input
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]

            face_shapes = ['Oval', 'Round', 'Square', 'Heart', 'Diamond']  # Adjust based on your model's classes
            predicted_shape = face_shapes[predicted_class]

            os.remove(filepath)  # Clean up the uploaded file
            return jsonify({'prediction': predicted_shape})

    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
