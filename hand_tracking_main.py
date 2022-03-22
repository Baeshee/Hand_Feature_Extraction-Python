import os

import cv2 as cv
import pandas as pd
from datetime import datetime as dt

from hand_detect_module import HandDetect

cap = cv.VideoCapture(0)
detector = HandDetect(detectCon=0.8, maxHands=2)
while True:
    succes, img = cap.read()
    if cv.waitKey(1) & 0xFF == ord('s'): 
        start_time = dt.now()
        hands, img = detector.staticImage(img)
        end_time = dt.now()
        duration = (end_time - start_time).total_seconds()
        
        i = 0
        path = "images/annoted_image"
        while os.path.exists(f"{path}_{i}.png"):
            i += 1
        cv.imwrite(f"{path}_{i}.png", img)
        
        dict = {}
        hTypes = []
        hScore = []
        lmNames = []
        lmIds = []
        xList = []
        yList = []
        zList = []
        
        if hands:
            for h in range(0, len(hands)):
                for v in range(0, len(hands[0]['lmList'])):
                    hTypes.append(hands[h]['type'])
                    hScore.append(hands[h]['score'])
                    lmNames.append(hands[h]['lmList'][v]['name'])
                    lmIds.append(hands[h]['lmList'][v]['id'])
                    xList.append(hands[h]['lmList'][v]['x_value'])
                    yList.append(hands[h]['lmList'][v]['y_value'])
                    zList.append(hands[h]['lmList'][v]['z_value'])        
        
        dict = {'hand_type': hTypes,
                'hand_score': hScore,
                'landmark_name': lmNames,
                'landmark_id': lmIds,
                'x_value': xList,
                'y_value': yList,
                'z_value': zList,
                'total_run_time': duration,
                'test_date': dt.now().strftime("%d-%m-%Y"),
                'test_time': dt.now().strftime("%H:%M:%S"),
                'annoted_image': f"{path}_{i}.png"}
        
        if os.path.exists("test_results.csv"):
            df = pd.read_csv("test_results.csv")
        else:
            df = pd.DataFrame()
        
        df = df.append(dict, ignore_index=True)
        df.to_csv('test_results.csv', index=True)
        
    # Display
    cv.imshow("Camera Capture", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv.destroyAllWindows()