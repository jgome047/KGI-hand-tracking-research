import mediapipe as mp
import cv2
import numpy as np
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

from matplotlib import pyplot as plt



#define the joints of interest (pretty much all of them)
joint_list = [[4,3,2], [8,7,6], [12,11,10], [16,15,14], [20,19,18]]

joint_list[3]


#function for calculating finger angles (for now, in 2D)
def draw_finger_angles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y, hand.landmark[joint[0]].z]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y, hand.landmark[joint[1]].z]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y, hand.landmark[joint[2]].z]) # Third coord
            
            #length of finger (might need to add some calcs distance formula sqrt((x2-x1)**2)+(y2-y1)**2))
            
            #len_ab = np.sqrt[(((b[0])-(a[0]))**2)+(((b[1])-(a[1]))**2)]
            #len_bc = np.sqrt[(((c[0])-(b[0]))**2)+(((c[1])-(b[1]))**2)]

            len_ab = 2.5
            len_bc = 2.5

            # assign easy to "read" coordinates based on arrays
            xa = a[0]
            ya = a[1]
            za = a[2]

            xb = b[0]
            yb = b[1]
            zb = b[2]

            # calculate the z position of point b
            zb = np.sqrt(len_ab**2 - (xb-xa)**2 - (yb-ya)**2)

            # assign easy to "read" coordinates based on array
            xc = c[0]
            yc = c[1]
            zc = c[2]

            # calculate the z position of point b
            zc = np.sqrt(len_bc**2 - (xc-xb)**2 - (yc-yb)**2)

            # calculate the length of segment ac
            len_ac = np.sqrt(xc**2 + yc**2 + zc**2)

            # compute the angle between segments ab and bc
            theta = np.arccos( (len_ab**2+len_bc**2-len_ac**2) / (2*len_ab*len_bc))
            
            angle = np.abs(theta*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
                
            cv2.putText(image, str(round(angle,2)), tuple(np.multiply([b[0],b[1]], 500).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image

#labelling hands as L and R, angles of each finger, and confidence values
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            
            # Process results
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{} {}'.format(label, round(score, 2))
            
            # Extract Coordinates
            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hands.HandLandmark.WRIST].x, hand.landmark[mp_hands.HandLandmark.WRIST].y)),
            [640,480]).astype(int))
            
            output = text, coords
            
    return output

#camera capture and drawing of finger segments
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Detections
        print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
                if get_label(num, hand, results):
                    text, coord = get_label(num, hand, results)
                    cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw angles to image from joint list
            draw_finger_angles(image, results, joint_list)
            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)



