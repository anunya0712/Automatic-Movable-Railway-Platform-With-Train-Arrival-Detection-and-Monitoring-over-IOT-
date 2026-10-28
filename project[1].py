import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

cap = cv2.VideoCapture(0) 
ripe_counts = [] 
unripe_counts = [] 

while True: 
    ret, frame = cap.read() 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    lower_ripe = np.array([0, 100, 100]) 
    upper_ripe = np.array([10, 255, 255]) 
    lower_unripe = np.array([35, 100, 100]) 
    upper_unripe = np.array([85, 255, 255]) 
    mask_ripe = cv2.inRange(hsv, lower_ripe, upper_ripe) 
    mask_unripe = cv2.inRange(hsv, lower_unripe, upper_unripe) 
    ripe_count = cv2.countNonZero(mask_ripe) 
    unripe_count = cv2.countNonZero(mask_unripe) 
    if ripe_count > unripe_count: 
        ripeness = "Ripe"
        text = f"Ripeness: {ripeness} "
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
    elif ripe_count < unripe_count: 
        ripeness = "Unripe" 
        text = f"Ripeness: {ripeness}" 
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
    else: 
        ripeness = "No fruit is detected" 
        text = f"Ripeness: {ripeness}" 
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        ripe_counts.append(ripe_count) 
        unripe_counts.append(unripe_count) 

    cv2.imshow('Fruit Ripeness Detector', frame) 

    if cv2.waitKey(1)==27: 
        break 

cap.release() 
cv2.destroyAllWindows() 

plt.plot(ripe_counts, label='Ripe') 
plt.plot(unripe_counts, label='Unripe') 
plt.ylabel('Count') 
plt.title('Fruit Ripeness Analysis') 
plt.legend() 
plt.show()

import cv2 
import numpy as np 
import mediapipe as mp

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Calculate angle using cosine rule
    angle = np.degrees(np.arccos(np.dot(b-a, c-a) / (np.linalg.norm(b-a) * np.linalg.norm(c-a))))
    return angle

# Video capture initialization
cap = cv2.VideoCapture(0)

# Counter initialization
counter = 0

# Stage initialization
stage = None

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read frame from video capture
        ret, frame = cap.read()
        frame=cv2.resize(frame,(700,700))
        if not ret:
            break

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Pose detection
        results = pose.process(image)

        # Convert frame back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Extract coordinates of shoulder, elbow, and wrist
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Display angle on frame
            cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Check for stage of exercise
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)

        except Exception as e:
            pass

        # Draw UI elements on frame
        cv2.rectangle(image, (0,0), (225,73), (245,117,16),-1)
        cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Draw landmarks on frame
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Display the frame
        cv2.imshow('Mediapipe Feed', image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()