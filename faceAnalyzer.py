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
