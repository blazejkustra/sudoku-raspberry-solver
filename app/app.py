import cv2
import numpy as np
import requests
import json


def preprocessing(img):
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img


def predict_digit(img):
    img = preprocessing(img)
    bytes_img = cv2.imencode(".png", img)[1].tobytes()
    response = requests.post(
        "http://127.0.0.1:5000/predict", data=bytes_img)
    prediction = json.loads(response.content)

    return prediction["digit_predicted"]


def make_image_rectangle(img):
    width = int(np.size(img, 1))
    height = int(np.size(img, 0))
    centre = abs(int((width - height) / 2))

    return img[0:height, centre:width - centre]


def process_image(img):
    blur_img = cv2.GaussianBlur(img, (5, 5), 1)
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 90, 20)

    return cv2.dilate(canny_img, np.ones((5, 5)), iterations=1)


def get_corner_points(dilate_img):
    max_area = 0
    sudoku_points = []

    contours, hierarchy = cv2.findContours(
        dilate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        points = cv2.approxPolyDP(contour, 50, 10, True)

        if area > max_area and len(points) == 4:
            sudoku_points = points
            max_area = area

    return sudoku_points


def sort_corner_points(points):
    if len(points) != 4:
        raise ValueError

    x_average = (points[0][0][0] + points[1][0][0] +
                 points[2][0][0] + points[3][0][0]) / 4
    y_average = (points[0][0][1] + points[1][0][1] +
                 points[2][0][1] + points[3][0][1]) / 4

    se = nw = ne = sw = None

    for point in points:
        point = point[0]
        if point[0] > x_average and point[1] > y_average:
            se = point
        elif point[0] < x_average and point[1] < y_average:
            nw = point
        elif point[0] > x_average and point[1] < y_average:
            ne = point
        elif point[0] < x_average and point[1] > y_average:
            sw = point

    return [nw, ne, sw, se]


def transform_perspective(img, points):
    before = np.float32(points)
    after = np.float32([[0, 0], [900, 0], [0, 900], [900, 900]])

    matrix = cv2.getPerspectiveTransform(before, after)
    return cv2.warpPerspective(img, matrix, (900, 900))


def separate_numbers_from_grid(transformed_img):
    blur_img = cv2.GaussianBlur(transformed_img, (3, 3), 1)
    gray_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray_img, 90, 20)
    dilate_img = cv2.dilate(canny_img, np.ones((6, 6)))
    processed_img = cv2.morphologyEx(
        dilate_img, cv2.MORPH_CLOSE, np.ones((6, 6)))

    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.morphologyEx(
        processed_img, cv2.MORPH_OPEN, horizontal_structure)
    vertical_img = cv2.morphologyEx(
        processed_img, cv2.MORPH_OPEN, vertical_structure)

    grid_img = cv2.dilate(horizontal_img + vertical_img, np.ones((15, 15)))

    return processed_img - grid_img


def get_numbers_contours(isolated_numbers_img):
    _, im_bw = cv2.threshold(isolated_numbers_img, 128, 255, 0)
    contours, _ = cv2.findContours(
        im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def predict_numbers(transformed_img, contours):
    sudoku_board = [[None] * 9 for _ in range(9)]
    height, width, _ = transformed_img.shape
    field_width = int(width / 9)
    field_height = int(height / 9)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_img = transformed_img[y:y + h, x:x + w]
        predicted_digit = predict_digit(digit_img)

        if predicted_digit is not None:
            y_index = int(np.floor(y / field_height))
            x_index = int(np.floor(x / field_width))
            sudoku_board[y_index][x_index] = predicted_digit

    return sudoku_board


def user_view(sudoku_board, user_img, contours):
    height, width, _ = user_img.shape
    field_width = int(width / 9)
    field_height = int(height / 9)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(user_img, (x, y),
                      (x+w, y+h), (0, 200, 0), 2)
        y_index = int(np.floor(y / field_height))
        x_index = int(np.floor(x / field_width))

        if sudoku_board[y_index][x_index]:
            cv2.putText(user_img, str(sudoku_board[y_index][x_index]), (x - 20, y),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return user_img


def read_image(img):
    processed_img = process_image(img)
    corner_points = get_corner_points(processed_img)

    try:
        corner_points = sort_corner_points(corner_points)
        transformed_img = transform_perspective(img, corner_points)
    except (ValueError, cv2.error):
        return [[None] * 9 for _ in range(9)]

    isolated_numbers_img = separate_numbers_from_grid(transformed_img)
    contours = get_numbers_contours(isolated_numbers_img)
    sudoku_board = predict_numbers(transformed_img, contours)
    user_img = user_view(sudoku_board, transformed_img, contours)

    return user_img, sudoku_board


def read_camera():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        cv2.imshow('result', img)

        if cv2.waitKey(1) == ord('q'):
            break

    success, img = cap.read()
    cap.release()

    cv2.imshow('last_frame', img)
    img = cv2.imread("./images/sudoku.jpg")  # TODO: change to img from webcam
    user_img, sudoku_board = read_image(img)

    cv2.imshow("result", user_img)
    print(sudoku_board)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_camera()
