import jetson.inference
import jetson.utils
import cv2

class NetModule():
    def __init__(self, path, threshold) -> None:
        self.path = path
        self.threshold = threshold
        
        # We load the network
        self.net = jetson.inference.detectNet(self.path, self.threshold)

        pass

    def detect(self, img, display = False):
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = self.net.Detect(imgCuda, overlay = "OVERLAY_NONE")

        objects = []
        # Method 2
        for d in detections:
            # print(d)
            # See DetectionObjectOutput.txt
            # Note (2)
            className = self.net.GetClassDesc(d.ClassID)
            objects.append([className, d])
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

    # Capture from camera at location 0.
    cap = cv2.VideoCapture(0)

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
