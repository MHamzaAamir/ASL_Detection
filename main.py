import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.pickle', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret,frame = cap.read()
        H = frame.shape[0]
        W = frame.shape[1]

        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgrgb)

        if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks:
            #     mp_drawing.draw_landmarks(
            #         frame,  # image to draw
            #         hand_landmarks,  # model output
            #         mp_hands.HAND_CONNECTIONS,  # hand connections
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):


                    data_aux.append(hand_landmarks.landmark[i].x - hand_landmarks.landmark[0].x)
                    data_aux.append(hand_landmarks.landmark[i].y - hand_landmarks.landmark[0].y)
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)


            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            pred_char = prediction[0]


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, pred_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    except:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)