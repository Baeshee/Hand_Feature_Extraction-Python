import os

import cv2 as cv
import pandas as pd
from datetime import datetime as dt

from hand_detect_module import HandDetect

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
detector = HandDetect(detectCon=0.8, maxHands=2)
while True:
    
    # Video capture and the action to start the analysis
    succes, img = cap.read()
    if cv.waitKey(1) & 0xFF == ord('s'): 
        
        # Detecting the hands of the person by using the camera feeder
        # combined with 2 timestamps to calculate the duration of the
        # processing request
        start_time = dt.now()
        hands, img = detector.staticImage(img)
        end_time = dt.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create a image where the landmarks are stored
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
        
        # Extracting all information that are stored in the variables
        # so they can be saved to a csv and excel for later use
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
                'landmark_id': lmIds,
                'landmark_name': lmNames,
                'x_value': xList,
                'y_value': yList,
                'z_value': zList,
                'total_run_time': duration,
                'test_date': dt.now().strftime("%d-%m-%Y"),
                'test_time': dt.now().strftime("%H:%M:%S"),
                'annoted_image': f"{path}_{i}.png"}
        
        # Check if there is a csv and a xslx file otherwise
        # create a new dataframe and save all the values to the
        # dataframe so they can be saved
        if os.path.exists("results/test_results.csv"):
            df = pd.read_csv("results/test_results.csv")
        else:
            df = pd.DataFrame()
        
        df_dict = pd.DataFrame(dict)
        df = pd.concat([df, df_dict], ignore_index=True)
        df.to_csv('results/test_results.csv', encoding='utf-8', index=False)
        # df.to_excel('results/test_results.xlsx', encoding='utf-8', index=False)
        
    # Display the camera view
    cv.imshow("Camera Capture", img)
    
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv.destroyAllWindows()