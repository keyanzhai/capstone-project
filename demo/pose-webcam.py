import cv2
import mediapipe as mp
import numpy as np
import rom

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    results = pose.process(image)

    if 'landmark' in dir(results.pose_world_landmarks):
        # right_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        # right_wrist_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
        # # right_wrist_pos = np.array([right_wrist.x, right_wrist.y])
        # print("right wrist depth: ", right_wrist_pos[2])

        # right_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        # right_elbow_pos = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
        # # right_elbow_pos = np.array([right_elbow.x, right_elbow.y])
        # print("right elbow depth: ", right_elbow_pos[2] * 100)

        # right_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        # # right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
        # right_shoulder_pos = np.array([right_shoulder.x  * image_width, right_shoulder.y * image_height, right_shoulder.z])
        # print("right shoulder depth: ", right_shoulder_pos[2] * 100)
        # rom_right_elbow = rom.rom_right_elbow(right_wrist_pos, right_elbow_pos, right_shoulder_pos)
        # print(rom_right_elbow)

        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        right_hip_pos = np.array([right_hip.x * image_width, right_hip.y  * image_height, right_hip.z])
        print("right hip x: ", right_hip_pos[0])

        # right_heel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
        # right_heel_pos = np.array([right_heel.x, right_heel.y, right_heel.z])
        # print("right heel x: ", right_heel_pos[0])





        

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
