import cv2
import mediapipe as mp
# import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

thumb_tip_x = 0
thumb_tip_y = 0
thumb_tip_z = 0
index_tip_x = 0
index_tip_y = 0
index_tip_z = 0
middle_tip_x = 0
middle_tip_y = 0
middle_tip_z = 0
ring_tip_x = 0
ring_tip_y = 0
ring_tip_z = 0
pinky_tip_x = 0
pinky_tip_y = 0
pinky_tip_z = 0

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():

        success, image = cap.read()

        # get size of image
        h, w, c = image.shape

        # start = time.time()

        # Filp the image for selfie-view display and convert the BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Set the image as reference to improve the performance
        image.flags.writeable = False

        # Process the image and get the hands
        results = hands.process(image)

        # Set the image as writable for drawing
        image.flags.writeable = True

        # Detect the hand
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 4:
                        thumb_tip_x = lm.x
                        thumb_tip_y = lm.y
                        thumb_tip_z = lm.z
                    if id == 8:
                        index_tip_x = lm.x
                        index_tip_y = lm.y
                        index_tip_z = lm.z
                    if id == 12:
                        middle_tip_x = lm.x
                        middle_tip_y = lm.y
                        middle_tip_z = lm.z
                    if id == 16:
                        ring_tip_x = lm.x
                        ring_tip_y = lm.y
                        ring_tip_z = lm.z
                    if id == 20:
                        pinky_tip_x = lm.x
                        pinky_tip_y = lm.y
                        pinky_tip_z = lm.z

                    # draw the center of cube
                    center_x = (thumb_tip_x + index_tip_x + middle_tip_x + ring_tip_x + pinky_tip_x) / 5
                    center_y = (thumb_tip_y + index_tip_y + middle_tip_y + ring_tip_y + pinky_tip_y) / 5
                    cv2.circle(image, (int(center_x*w), int(center_y*h)), 5, (0, 0, 255), -1)

        # end = time.time()
        # totalTime = end - start
        # fps = 1 / totalTime

        # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()