import cv2
import mediapipe as mp
import numpy as np
import math
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

lambda_n = 0.2
correction_factor = 1

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
                # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) # Draw the hand landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 4:
                        thumb_tip_x = lm.x
                        thumb_tip_y = lm.y
                        thumb_tip_z = lm.z * correction_factor
                    if id == 8:
                        index_tip_x = lm.x
                        index_tip_y = lm.y
                        index_tip_z = lm.z * correction_factor
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
                        pinky_tip_z = lm.z * correction_factor

                    # calculate the position of the center, based on the average of the tip of the fingers
                    center_x = (thumb_tip_x * 3 + index_tip_x + middle_tip_x + ring_tip_x + pinky_tip_x) / 7
                    center_y = (thumb_tip_y * 3 + index_tip_y + middle_tip_y + ring_tip_y + pinky_tip_y) / 7
                    center_z = (thumb_tip_z * 3 + index_tip_z + middle_tip_z + ring_tip_z + pinky_tip_z) / 7
                    center = np.array([[center_x, center_y, center_z]])

                    # draw the cube
                    # 1. calculate the spatial direction of cube
                    a = (index_tip_y - thumb_tip_y) * (pinky_tip_z - thumb_tip_z) - (pinky_tip_y - thumb_tip_y) * (index_tip_z - thumb_tip_z)
                    b = (index_tip_z - thumb_tip_z) * (pinky_tip_x - thumb_tip_x) - (pinky_tip_z - thumb_tip_z) * (index_tip_x - thumb_tip_x)
                    c = (index_tip_x - thumb_tip_x) * (pinky_tip_y - thumb_tip_y) - (pinky_tip_x - thumb_tip_x) * (index_tip_y - thumb_tip_y)
                    length = (a**2 + b**2 + c**2)**0.5+1e-9
                    lambda_n = 1
                    normal_vector = np.array([[a/length, b/length, c/length]])

                    # 2. draw the 8 points
                    # 2.1 calculate spatial rotation angle
                    x_axe = np.array([[1, 0, 0]])
                    y_axe = np.array([[0, 1, 0]])
                    z_axe = np.array([[0, 0, 1]])
                    cos_omega = normal_vector.dot(y_axe.T) / ((np.linalg.norm(normal_vector) * np.linalg.norm(y_axe)) + 1e-9)
                    sin_omega = math.sin(math.acos(cos_omega))
                    cos_phi = normal_vector.dot(z_axe.T) / ((np.linalg.norm(normal_vector) * np.linalg.norm(z_axe)) + 1e-9)
                    sin_phi = math.sin(math.acos(cos_phi))
                    cos_kappa = normal_vector.dot(x_axe.T) / ((np.linalg.norm(normal_vector) * np.linalg.norm(x_axe)) + 1e-9)
                    sin_kappa = math.sin(math.acos(cos_kappa))
                    r11 = cos_phi*cos_kappa
                    r12 = -cos_phi*sin_kappa
                    r13 = sin_phi
                    r21 = cos_omega*sin_kappa + sin_omega*sin_phi*cos_kappa
                    r22 = cos_omega*cos_kappa - sin_omega*sin_phi*sin_kappa
                    r23 = -sin_omega*cos_phi
                    r31 = sin_omega*sin_kappa - cos_omega*sin_phi*cos_kappa
                    r32 = sin_omega*cos_kappa + cos_omega*sin_phi*sin_kappa
                    r33 = cos_omega*cos_phi
                    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

                    # 2.2 create points, translate them and rotate them
                    scale_factor = length
                    T_scale = np.array([[scale_factor, 0, 0, 0], [0, scale_factor, 0, 0], [0, 0, scale_factor, 0], [0, 0, 0, 1]])
                    T_trans_part = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
                    top_00 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[lambda_n, lambda_n, lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    top_00 = R.dot(top_00[:3])
                    cv2.circle(image, (int((center_x+top_00[0])*w), int((center_y+top_00[1])*h)), 5, (0, 0, 255), -1)
                    top_01 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[lambda_n, -lambda_n, lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    top_01 = R.dot(top_01[:3])
                    cv2.circle(image, (int((center_x+top_01[0])*w), int((center_y+top_01[1])*h)), 5, (0, 0, 255), -1)
                    top_10 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[-lambda_n, lambda_n, lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    top_10 = R.dot(top_10[:3])
                    cv2.circle(image, (int((center_x+top_10[0])*w), int((center_y+top_10[1])*h)), 5, (0, 0, 255), -1)
                    top_11 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[-lambda_n, -lambda_n, lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    top_11 = R.dot(top_11[:3])
                    cv2.circle(image, (int((center_x+top_11[0])*w), int((center_y+top_11[1])*h)), 5, (0, 0, 255), -1)
                    bottom_00 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[lambda_n, lambda_n, -lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    bottom_00 = R.dot(bottom_00[:3])
                    cv2.circle(image, (int((center_x+bottom_00[0])*w), int((center_y+bottom_00[1])*h)), 5, (0, 0, 255), -1)
                    bottom_01 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[lambda_n, -lambda_n, -lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    bottom_01 = R.dot(bottom_01[:3])
                    cv2.circle(image, (int((center_x+bottom_01[0])*w), int((center_y+bottom_01[1])*h)), 5, (0, 0, 255), -1)
                    bottom_10 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[-lambda_n, lambda_n, -lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    bottom_10 = R.dot(bottom_10[:3])
                    cv2.circle(image, (int((center_x+bottom_10[0])*w), int((center_y+bottom_10[1])*h)), 5, (0, 0, 255), -1)
                    bottom_11 = T_scale.dot(np.concatenate((T_trans_part, np.concatenate((np.array([[-lambda_n, -lambda_n, -lambda_n]]).T, np.array([[1]])))), axis=1).dot(np.array([[0], [0], [0], [1]])))
                    bottom_11 = R.dot(bottom_11[:3])
                    cv2.circle(image, (int((center_x+bottom_11[0])*w), int((center_y+bottom_11[1])*h)), 5, (0, 0, 255), -1)

                    # 3. draw the cube
                    cv2.line(image, (int((center_x+top_00[0])*w), int((center_y+top_00[1])*h)), (int((center_x+top_01[0])*w), int((center_y+top_01[1])*h)), (255, 255, 255), 2)
                    cv2.line(image, (int((center_x+top_01[0])*w), int((center_y+top_01[1])*h)), (int((center_x+top_11[0])*w), int((center_y+top_11[1])*h)), (255, 255, 255), 2)
                    cv2.line(image, (int((center_x+top_11[0])*w), int((center_y+top_11[1])*h)), (int((center_x+top_10[0])*w), int((center_y+top_10[1])*h)), (255, 255, 255), 2)
                    cv2.line(image, (int((center_x+top_10[0])*w), int((center_y+top_10[1])*h)), (int((center_x+top_00[0])*w), int((center_y+top_00[1])*h)), (255, 255, 255), 2)
                    cv2.line(image, (int((center_x+bottom_00[0])*w), int((center_y+bottom_00[1])*h)), (int((center_x+bottom_01[0])*w), int((center_y+bottom_01[1])*h)), (150, 150, 150), 2)
                    cv2.line(image, (int((center_x+bottom_01[0])*w), int((center_y+bottom_01[1])*h)), (int((center_x+bottom_11[0])*w), int((center_y+bottom_11[1])*h)), (150, 150, 150), 2)
                    cv2.line(image, (int((center_x+bottom_11[0])*w), int((center_y+bottom_11[1])*h)), (int((center_x+bottom_10[0])*w), int((center_y+bottom_10[1])*h)), (150, 150, 150), 2)
                    cv2.line(image, (int((center_x+bottom_10[0])*w), int((center_y+bottom_10[1])*h)), (int((center_x+bottom_00[0])*w), int((center_y+bottom_00[1])*h)), (150, 150, 150), 2)
                    cv2.line(image, (int((center_x+top_00[0])*w), int((center_y+top_00[1])*h)), (int((center_x+bottom_00[0])*w), int((center_y+bottom_00[1])*h)), (0, 0, 0), 2)
                    cv2.line(image, (int((center_x+top_01[0])*w), int((center_y+top_01[1])*h)), (int((center_x+bottom_01[0])*w), int((center_y+bottom_01[1])*h)), (0, 0, 0), 2)
                    cv2.line(image, (int((center_x+top_11[0])*w), int((center_y+top_11[1])*h)), (int((center_x+bottom_11[0])*w), int((center_y+bottom_11[1])*h)), (0, 0, 0), 2)
                    cv2.line(image, (int((center_x+top_10[0])*w), int((center_y+top_10[1])*h)), (int((center_x+bottom_10[0])*w), int((center_y+bottom_10[1])*h)), (0, 0, 0), 2)


        # end = time.time()
        # totalTime = end - start
        # fps = 1 / totalTime

        # cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Cube in hand', image)

        # quit the program with the esc key
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()