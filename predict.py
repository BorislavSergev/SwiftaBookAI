import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img
import numpy as np

def predict_image(file):
    try:
        # Load model
        model = tf.keras.models.load_model('model.h5')

        # Load and preprocess image
        img = load_img(file, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        return {'predicted_class': int(predicted_class)}

    except Exception as e:
        return {'error': str(e)}
