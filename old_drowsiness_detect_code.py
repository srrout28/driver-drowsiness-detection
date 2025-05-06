import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils
import imutils
#import numpy as np
import pygame
#import time

# Initialize pygame for alerting sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.mpeg')

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Main function for drowsiness detection
def drowsiness_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ear_threshold = 0.30
    consecutive_frames = 35
    counter = 0

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        face_rectangles = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_rectangles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        for face in faces:
            landmarks = predictor(gray, face)
            shape = face_utils.shape_to_np(landmarks)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEyeEAR = eye_aspect_ratio(leftEye)
            rightEyeEAR = eye_aspect_ratio(rightEye)
            ear = (leftEyeEAR + rightEyeEAR) / 2

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)

            if ear < ear_threshold:
                counter += 1
                cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if counter >= consecutive_frames:
                    pygame.mixer.music.play(-1)
                    cv2.putText(frame, "*****DROWSINESS ALERT!****", (170, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "---WAKE UP---", (200, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
            else:
                counter = 0
                pygame.mixer.music.stop()
                cv2.putText(frame, "Eyes Open", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "BSc.(CS) Project, UN (Auto) College of Sc. & Tech.", (3, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (153, 255, 102), 1)

        cv2.imshow("Frames", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    drowsiness_detection()
