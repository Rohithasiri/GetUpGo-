import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def run_crunches(user_weight,video_path=0 , reps=10):
    st.title("💪 Crunches Tracker")
    st.markdown(f"**Target Reps:** {reps}")
    stframe = st.empty()

    counter = 0
    stage = None
    calories_per_rep = 0.35  # average for crunches
    calories_burned = 0
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            h, w, _ = frame.shape
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                            lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                       lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
                knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]

                angle = calculate_angle(shoulder, hip, knee)

                if angle > 120:
                    stage = "down"
                if angle < 110 and stage == "down":
                    stage = "up"
                    counter += 1
                    calories_burned = round(counter * calories_per_rep, 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            elapsed_time = round(time.time() - start_time, 1)
            cv2.rectangle(image, (0, 0), (320, 100), (0, 0, 0), -1)
            cv2.putText(image, f'Crunches: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # cv2.putText(image, f'Calories: {calories_burned}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(image, f'Time: {elapsed_time}s', (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            stframe.image(image, channels="BGR")

            if counter >= reps:
                st.success(f"🎉 Goal Achieved! {counter} crunches done.")
                break

    cap.release()
    cv2.destroyAllWindows()
