# https://www.ebenezertechs.com/mobilenet-ssd-using-opencv-3-4-1-deep-learning-module-python/
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

import numpy as np
from cv2 import cv2
import argparse
import argparse
import imutils
import time
from imutils.video import VideoStream, FileVideoStream, FPS
import sys
from queue import Queue

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
help="path to caffe 'deploy' model")
ap.add_argument("-m", "--model", required=True,
help="path to caffe pre trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=True,
help="path to input video file")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


print("[INFO] starting video stream ...")
vs = cv2.VideoCapture(args["video"])
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))
time.sleep(2.0)
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = vs.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = np.dstack([frame, frame, frame])
    # frame = imutils.resize(frame, width=300)
    
    # blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
	# (300, 300), 127.5)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 
    (127.5, 127.5, 127.5), False)

    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < args["confidence"]:
            continue
        
        (h, w) = frame.shape[:2]
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
        print("[INFO] {}".format(label))
        cv2.rectangle(frame, (startX, startY), (endX, endY),
        COLORS[idx], 2)

        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    out.write(frame)    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

out.release()
cv2.destroyAllWindows()
vs.stop()