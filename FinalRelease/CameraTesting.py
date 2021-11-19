# Import needed libraries
import cv2
import NetNTTSModule as nemou
from gtts import gTTS
import os

# Capture from camera
cap = cv2.VideoCapture('/dev/video0')

# Set the width and height of the frames in the video stream.
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# We create an object
myModel = nemou.NetModule("ssd-mobilenet-v2", 0.5)

while True:
    success, img = cap.read()
    objects = myModel.detect(img, True)
    cv2.imshow("Object Detection | Network", img)
    cv2.moveWindow("Object Detection | Network", 0, 0)
    if (cv2.waitKey(1) == ord('q')):
        goodbyeMsg = gTTS(text = "Goodbye!", lang = 'en', slow = False)
        goodbyeMsg.save('output.mp3')
        os.system('mpg123 output.mp3')
        break

cap.release()
cv2.destroyAllWindows()
    
