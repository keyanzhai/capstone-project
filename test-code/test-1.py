import cv2
import mediapipe as mp
import numpy as np
import time

start_time = time.time()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  
  leftStart_y = 0
  rightStart_y = 0
  leftStart_x = 0
  rightStart_x = 0
  rightPrev_x = 0
  leftPrev_x = 0
  frame = 0
  stand_state = 0 # 0 for sitting, 1 for standing
  last_stand_state = 0 # last stand_state
  rightx_distance = 0
  leftx_distance = 0

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
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_shoulder_pos = np.array([right_shoulder.x * image_width, right_shoulder.y  * image_height, right_shoulder.z])
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_shoulder_pos = np.array([left_shoulder.x * image_width, left_shoulder.y  * image_height, left_shoulder.z])
        rightCurr_x = right_shoulder_pos[0]
        rightCurr_y = right_shoulder_pos[1]
        leftCurr_x = left_shoulder_pos[0]
        leftCurr_y = left_shoulder_pos[1]

        if (frame == 1):
          rightStart_y = rightCurr_y
          rightStart_x = rightCurr_x

          leftStart_y = leftCurr_y
          leftStart_x = leftCurr_x

          rightPrev_x = rightStart_x
          leftPrev_x = leftStart_x

        rightx_distance += abs(rightCurr_x - rightPrev_x)
        leftx_distance += abs(leftCurr_x - leftPrev_x)
        # Print out left and right shoulder distance
        print("Right Shoulder Distance = ", rightx_distance)
        print("Left Shoulder Distance = ", leftx_distance)

        if abs(rightCurr_y - rightStart_y) < image_height * 0.2 and abs(leftCurr_y - leftStart_y) < image_height * 0.2:
          stand_state = 0
          if abs(rightCurr_y - rightStart_y) < image_height * 0.05 and abs(leftCurr_y - leftStart_y) < image_height * 0.05 and (rightx_distance > image_width * 0.4) and (leftx_distance > image_width * 0.4):
            total_time = time.time() - start_time
            print("total_time = ", total_time)
            # break
          print("Sitting")
        else:
          stand_state = 1
          print("Standing")

        last_stand_state = stand_state
        leftPrev_x = leftCurr_x
        rightPrev_x = rightCurr_x        

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    frame += 1

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap.release()
