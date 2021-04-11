import requests
import cv2
import numpy as np
import json


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = cv2.resize(img, (32, 32))
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def predict_test():
    for number in range(1, 10):
        img = cv2.imread(f"src/tests/numbers/{number}.png")
        img = preprocessing(img)
        img_bytes = cv2.imencode(".png", img)[1].tobytes()
        response = requests.post(
            "http://127.0.0.1:5000/predict", data=img_bytes)
        prediction = json.loads(response.content)
        digit_predicted = prediction["digit_predicted"]
        assert digit_predicted == number, f"Should be {number} but received {digit_predicted}"


if __name__ == "__main__":
    predict_test()
