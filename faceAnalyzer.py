import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self, minDetectionCon= 0.5):

        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        # self.faceDetection = self.mpFaceDetection.FaceDetection(0.75)
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
    
    def findFaces(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                # self.mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img, bbox)
                    # cv2.rectangle(img, bbox, (80, 255, 0), 2)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%',
                            (bbox[0],bbox[1]-20),
                            cv2.FONT_HERSHEY_PLAIN,
                            3, (0, 255, 0), 2)
        return img, bboxs

    def fancyDraw(self, img, bbox, l = 30, thickness = 5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (80, 255, 0), 1)
        # Top Left x,y
        cv2.line(img, (x,y),(x+l,y),(80,255,0),thickness)
        cv2.line(img, (x, y), (x, y + l), (80, 255, 0), thickness)
        # Top Right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (80, 255, 0), thickness)
        cv2.line(img, (x1, y), (x1, y + l), (80, 255, 0), thickness)
        # Bottom Left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (80, 255, 0), thickness)
        cv2.line(img, (x, y1), (x, y1 - l), (80, 255, 0), thickness)
        # Bottom Right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (80, 255, 0), thickness)
        cv2.line(img, (x1, y1), (x1, y1 - l), (80, 255, 0), thickness)

        return img
