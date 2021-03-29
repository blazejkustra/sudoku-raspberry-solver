######## Webcam hand-digit Detection Using Tensorflow mnist Model #########
#
# Author: jihwan Lee
# Date: 02/10/20
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam feed
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='mnist.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Import TensorFlow libraries
# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

def DED(grayImg):
    minInt, maxInt, minLoc, maxLoc = cv2.minMaxLoc(grayImg)
    beam = cv2.mean(grayImg)
    mean = float(beam[0])
    CannyOfTuna = cv2.Canny(grayImg, (mean + minInt)/2, (mean + maxInt)/2)
    return CannyOfTuna

# Initialize video stream
#### videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
#### time.sleep(1)

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    #### frame1 = videostream.read()
    frame1 = cv2.imread("./sudoku.png")
    frame1 = cv2.resize(frame1, (900, 900), interpolation = cv2.INTER_AREA)
    # Convert RGB to GRAY
    frame = frame1.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, frame_gray) = cv2.threshold(255 - frame_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Get contours in image
    _, ctrs, hier = cv2.findContours(DED(frame_gray.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles around all contours
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    for rect in rects:
        rect_area = rect[2] * rect[3]
        img_area = imW * imH

        area_ratio = rect_area / img_area
        # if rect is too small or too big, ignore the rect
        #print("area_ratio:" + str(area_ratio))
        #### if area_ratio <= 0.01 or area_ratio >= 0.3:
        if area_ratio >= 0.3:
            continue
        

        # draw rectangle
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        leng = int(rect[3] * 1.6)
        pt1 = abs(int(rect[1] + rect[3] // 2 - leng // 2))
        pt2 = abs(int(rect[0] + rect[2] // 2 - leng // 2))

        # extract rect in gray-scale image
        roi = frame_gray[pt1:pt1+leng, pt2:pt2+leng]

        try:
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            input_data = np.expand_dims(roi, axis=0)
            input_data = input_data.reshape(1, 28, 28, 1)

            if floating_model:
                input_data = np.float32(input_data) / 255.0

            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

            output_details = interpreter.get_output_details()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    object_name = labels[np.argmax(output_data)] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: '7: 60%'
                    cv2.putText(frame, label, (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 3)

            roi = cv2.dilate(roi, (3, 3))
        except:
            continue


    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Digit detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
#videostream.stop()
