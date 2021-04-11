from flask import jsonify
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("../model/model.h5")


def predict_number(img_bytes):
    try:
        # decode into an numpy array
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)

        # transform in order to fit into the model
        batch = np.expand_dims(img, 0).astype(np.float32)
        img_for_prediction = np.expand_dims(batch, axis=3)

        # make a prediction
        predictions = model.predict(img_for_prediction)
        digit_predicted = np.argmax(predictions[0]) + 1
        return {"digit_predicted": int(digit_predicted)}
    except ValueError as e:
        print(e)
        return {"digit_predicted": None}
