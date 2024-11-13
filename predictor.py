"""
FelipedelosH


0 - it not found model-date.keras file you need execute: 0_create_train_neuralnetwork.py
1 - select a image in function 'predict_image(path)'
3 - execute script and the result show in terminal.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


_modelName = "model-2024-11-12-21.10.keras"
model = tf.keras.models.load_model(_modelName)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Redimensiona a 128x128
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0] < 0.5:
        print("Es un gato 🐱")
    else:
        print("Es un perro 🐶")

# Images to Predict
predict_image("0.jpg")
