import mediapipe as mp
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import serial

# Set the serial port for Arduino communication
port = 'COM14'
arduino = serial.Serial(port=port, baudrate=9600, timeout=0.01)


# Function to set servo angles based on finger angles
def set_angles(angles):

    msg = ''
    
    # Prepare the message for Arduino
    for angle in angles:
        a = str(angle)
        while len(a) < 3:
            a = '0' + a
        msg += a
    
    msg = '<' + msg + '>'

    # Send the message to Arduino
    print("Sending: ", msg)    
    for c in msg:
        arduino.write(bytes(c, 'utf-8'))

    # Receive and print the response from Arduino
    data = arduino.readline()
    print("Receiving: ", data)


# Function to translate a value from one range to another
def translate(value, leftMin, leftMax, rightMin, rightMax):

    # Figure out how the size of each range
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Normalize the value 
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Scale the value 
    return rightMin + (valueScaled * rightSpan)


# Function to compute finger angles based on hand landmarks
def compute_finger_angles(image, results, joint_list):

    angles = []

    # Iterate through detected hands
    for hand in results.multi_hand_landmarks:
        for i, joint in enumerate(joint_list):
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

            rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(rad*180.0/np.pi)

            if angle > 65:
                angle = 360 - angle
            
            # Interpolate and limit the angle values
            if i == 0:
                angle = np.interp(angle,[90,180],[0, 200])
                angle = min(65, angle)
            else:
                angle = np.interp(angle,[30,180],[0, 180])
                angle = min(65, angle)

            angles.append(int(angle))

            # Display the angle on the image
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2, cv2.LINE_AA)
    return image, angles


# Initialize MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open the webcam
cap = cv2.VideoCapture(0)

# Define joints for finger tracking
joint_list = [ [4, 3, 2], [7, 6, 5], [11, 10, 9], [15, 14, 13], [19, 18, 17]]


# Start hand tracking
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():

        # Read a frame from the webcam
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip the image horizontally for a more natural view
        image = cv2.flip(image, 1)
        
        # Process the image with MediaPipe hands module
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Render hand landmarks and finger connections
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

            # Compute finger angles and set them using Arduino communication
            image, angles = compute_finger_angles(image, results, joint_list)
            set_angles(angles)
        
        # Convert the image back to BGR for displaying
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', image)

        # Exit the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
