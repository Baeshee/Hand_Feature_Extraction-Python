import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

IMAGE_FILES = ['D:\School\Jaar 3\IPMEDTH\Python Code\hand_5.jpg']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
    ) as hands:
    for i, file in enumerate(IMAGE_FILES):
        image=cv.flip(cv.imread(file), 1)
        results = hands.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        
        print('Handedness: ', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print('hand_landmarks: ', hand_landmarks)
            print(
                f'Index finger tip coordinates: (',
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            )
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            cv.imwrite(
                'annotated_image' + str(i) + '.png', cv.flip(annotated_image, 1))
        # Draw hand world landmarks.
            if not results.multi_hand_world_landmarks:
                continue
            for hand_world_landmarks in results.multi_hand_world_landmarks:
                mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)