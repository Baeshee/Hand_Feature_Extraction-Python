import cv2 as cv
import mediapipe as mp
from datetime import datetime as dt

tipIDS = [4, 8, 12, 16, 20]

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
mpDrawStyle = mp.solutions.drawing_styles

while True:
    succes, img = cap.read()
    imageRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    if cv.waitKey(1) & 0xFF == ord('s'): 
        first_time = dt.now()
        imageFlip = cv.flip(imageRGB, 1)
        results = hands.process(imageFlip)
        print('Handedness: ', results.multi_handedness)  
        
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = imageFlip.shape
        annotated_image = imageFlip.copy() 
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                print(results.multi_handedness[0].classification[0].label)
                print(id)
                print(lm)
            mpDraw.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mpHands.HAND_CONNECTIONS,
                mpDrawStyle.get_default_hand_landmarks_style(),
                mpDrawStyle.get_default_hand_connections_style())
            cv.imwrite(
                'annotated_image.png', cv.flip(annotated_image, 1))
        
        second_time = dt.now()
        print((second_time - first_time).total_seconds())
        
    cv.imshow('frame', img)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break