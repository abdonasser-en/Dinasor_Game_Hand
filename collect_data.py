import os
import time
import random
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

list_l = []
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                a = hand_landmarks.landmark
                x = []
                y = []
                z = []
                for p in a:
                    x.append(p.x)
                    y.append(p.y)
                    z.append(p.z)

                list_l.append([np.mean(x), np.mean(y), np.mean(z), np.std(x), np.std(y),1])
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.resize(
            cv2.flip(image, 1), (360, 240)))
        if cv2.waitKey(5) & 0xFF == 27:
            df = pd.DataFrame(list_l, index=None,
                  columns=['meanX', 'meanY', 'meanZ', 'stdX', 'stdY','class'])
            df.to_csv("class1_data.csv")
            break

cap.release()
