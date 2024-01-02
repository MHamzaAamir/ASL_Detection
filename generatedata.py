import cv2 
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


data_dir = './data'

data = []
labels = []

count = 0
for dir in os.listdir(data_dir):
    for img_path in os.listdir(os.path.join(data_dir,dir)):

        temp = []
        img = cv2.imread(os.path.join(data_dir,dir,img_path))
        imgrgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = hands.process(imgrgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):

                    x = hand_landmarks.landmark[i].x - hand_landmarks.landmark[0].x
                    y = hand_landmarks.landmark[i].y - hand_landmarks.landmark[0].y
                    temp.append(x)
                    temp.append(y)
            
            data.append(temp)
            labels.append(dir)

print(len(labels))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()