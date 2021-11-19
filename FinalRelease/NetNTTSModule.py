import jetson.inference
import jetson.utils
import cv2
import numpy as np 
import time
import os
from gtts import gTTS
import threading

item = "Welcome! Object Detection starts now."
itemAux = ""
speak = True

class NetModule():
    def __init__(self, path, threshold) -> None:
        self.path = path
        self.threshold = threshold
        
        # We load the network
        self.net = jetson.inference.detectNet(self.path, self.threshold)

        pass

    def sayItem():
        global speak
        global item
        while True:
            if speak == True:
                output = gTTS(text = item, lang = 'en', slow = False)
                output.save('output.mp3')
                os.system('mpg123 output.mp3')
                speak = False

    x = threading.Thread(target = sayItem, daemon = True)
    x.start()

    def detect(self, img, display = False):
        global speak, item, itemAux
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = self.net.Detect(imgCuda, overlay = "OVERLAY_NONE")
        objects = []
        # Method 2
        for d in detections:
            if speak == False:
                confidence = round(d.Confidence, 3)
                if confidence >= .9:
                    item = self.net.GetClassDesc(d.ClassID)
                    if item != itemAux:
                        speak = True
                else:
                    item = ""
                itemAux = item
            # See DetectionObjectOutput.txt
            # Note (2)
            objects.append([item, d])
            # Note (1)
            if display:
                # Coordinates of the detected container
                # x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                # Setting up OpenCV for our image result
                # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f'FPS: {int(self.net.GetNetworkFPS())}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 191, 0), 2)
                cv2.putText(img, objects[0][0], (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 64), 2)

        return objects

def main():

    # Capture from camera.
    cap = cv2.VideoCapture('/dev/video0')

    # Set the width and height of the frames in the video stream.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # We create an object
    myModel = NetModule("ssd-mobilenet-v2", 0.5)

    while True:
        success, img = cap.read()
        objects = myModel.detect(img)
        if len(objects) != 0:
            print(objects[0][0])

        # Method 1 (NO FLEXIBILITY)
        # We convert Cuda image back to Numpy.
        # img = jetson.utils.cudaToNumpy(imgCuda)

        cv2.imshow("Object Detection Camera", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
