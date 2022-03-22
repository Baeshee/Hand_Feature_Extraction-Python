import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pointsToTrack = [0, 17, 18, 19, 20]

while True:
    success, image = cap.read()
    imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # if id == 20:
                #     cv.circle(image, (cx, cy), 25, (255, 0, 255), cv.FILLED)
                
                print(id)
                print(lm)
                
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            
    cv.imshow("Output", image)
    cv.waitKey(1)
                    