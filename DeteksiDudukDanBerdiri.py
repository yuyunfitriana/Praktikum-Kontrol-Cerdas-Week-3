import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def hitung_sudut(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba)*np.linalg.norm(bc))
    sudut = np.degrees(np.arccos(cosine_angle))
    return sudut

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,results.pose_landmarks,
mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        sudut_kaki = hitung_sudut(hip,knee,ankle)

        if sudut_kaki > 160:
            status = "BERDIRI"
        elif sudut_kaki < 120:
            status = "DUDUK"
        else:
            status = "TRANSISI"

        cv2.putText(frame,status, (50,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
    cv2.imshow("Deteksi Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()