# hands
# John Gomez
import cv2
import mediapipe as mp
import numpy as np
import scipy

from scipy.spatial import distance


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
import time

## For static images:
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.imread('./Hands - 100/Hand_0000002.jpg')  # Insert your Image Here
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        print("Continue")

    # Print handedness and draw hand landmarks on the image.
    # print('Handedness:', results.multi_handedness)
    image_height, image_width, _ = image.shape
    # annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:

        # thumb calculations
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
        thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
        thumb_tip = [thumb_tip_x, thumb_tip_y]

        thumb_ip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
        thumb_ip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
        thumb_ip = [thumb_ip_x, thumb_ip_y]

        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
        thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
        thumb_mcp = [thumb_mcp_x, thumb_mcp_y]

        thumb_cmc_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
        thumb_cmc_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
        thumb_cmc = [thumb_cmc_x, thumb_cmc_y]

        # get all coordinates of 4 points of finger, get distance between all points and convert to inches
        thumb_length = (
                (distance.euclidean(thumb_tip, thumb_ip)) + (distance.euclidean(thumb_ip, thumb_mcp)) + \
                (distance.euclidean(thumb_mcp, thumb_cmc))) * 0.010417

        print(thumb_length, "in")

        # index calculations

        index_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
        index_tip = [index_tip_x, index_tip_y]

        index_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
        index_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
        index_dip = [index_dip_x, index_dip_y]

        index_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
        index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
        index_pip = [index_pip_x, index_pip_y]

        index_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
        index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
        index_mcp = [index_mcp_x, index_mcp_y]

        index_length = ((distance.euclidean(index_tip, index_dip)) + (distance.euclidean(index_dip, index_pip)) + \
                        (distance.euclidean(index_pip, index_mcp))) * 0.010417

        print(index_length, "in")

        # middle calculations

        middle_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
        middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
        middle_tip = [middle_tip_x, middle_tip_y]

        middle_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
        middle_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
        middle_dip = [middle_dip_x, middle_dip_y]

        middle_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
        middle_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
        middle_pip = [middle_pip_x, middle_pip_y]

        middle_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
        middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
        middle_mcp = [middle_mcp_x, middle_mcp_y]

        middle_length = ((distance.euclidean(middle_tip, middle_dip)) + (distance.euclidean(middle_dip, middle_pip)) + \
                        (distance.euclidean(middle_pip, middle_mcp))) * 0.010417

        print(middle_length, "in")

        # ring calculations

        ring_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
        ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
        ring_tip = [ring_tip_x, ring_tip_y]

        ring_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
        ring_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
        ring_dip = [ring_dip_x, ring_dip_y]

        ring_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
        ring_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
        ring_pip = [ring_pip_x, ring_pip_y]

        ring_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
        ring_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
        ring_mcp = [ring_mcp_x, ring_mcp_y]

        ring_length = ((distance.euclidean(ring_tip, ring_dip)) + (distance.euclidean(ring_dip, ring_pip)) + \
                         (distance.euclidean(ring_pip, ring_mcp))) * 0.010417

        print(ring_length, "in")

        # pinky calculations

        pinky_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
        pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
        pinky_tip = [pinky_tip_x, pinky_tip_y]

        pinky_dip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
        pinky_dip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
        pinky_dip = [pinky_dip_x, pinky_dip_y]

        pinky_pip_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
        pinky_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
        pinky_pip = [pinky_pip_x, pinky_pip_y]

        pinky_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
        pinky_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
        pinky_mcp = [pinky_mcp_x, pinky_mcp_y]

        pinky_length = ((distance.euclidean(pinky_tip, pinky_dip)) + (distance.euclidean(pinky_dip, pinky_pip)) + \
                       (distance.euclidean(pinky_pip, pinky_mcp))) * 0.010417

        print(pinky_length, "in")






        # mp_drawing.draw_landmarks(
            # annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # cv2.imwrite(r'Hand_0000002.png', annotated_image)
