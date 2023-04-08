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
  

    start_pos = np.zeros((6,3));
    visibility = np.zeros((6,1));
    frame = 0
    count = 0
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
            # heel
            right_heel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
            right_heel_pos = np.array([right_heel.x * image_width, right_heel.y * image_height, right_heel.z * image_width])
            left_heel = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            left_heel_pos = np.array([left_heel.x * image_width, left_heel.y * image_height, left_heel.z * image_width])

            # ankle
            right_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_ankle_pos = np.array([right_ankle.x * image_width, right_ankle.y * image_height, right_ankle.z * image_width])
            left_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            left_ankle_pos = np.array([left_ankle.x * image_width, left_ankle.y * image_height, left_ankle.z * image_width])

            # index
            right_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            right_index_pos = np.array([right_index.x * image_width, right_index.y * image_height, right_index.z * image_width])
            left_index = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            left_index_pos = np.array([left_index.x * image_width, left_index.y * image_height, left_index.z * image_width])

            if (frame >= 30):
                if (frame == 30):
                    start_pos[0] = right_heel_pos
                    start_pos[1] = right_ankle_pos
                    start_pos[2] = right_index_pos
                    start_pos[3] = left_heel_pos
                    start_pos[4] = left_ankle_pos
                    start_pos[5] = left_index_pos
                    start_time = time.time()
                
                curr_pos = np.zeros((6,3))
                curr_pos[0] = right_heel_pos
                curr_pos[1] = right_ankle_pos
                curr_pos[2] = right_index_pos
                curr_pos[3] = left_heel_pos
                curr_pos[4] = left_ankle_pos
                curr_pos[5] = left_index_pos

                # Visibility matrix
                visibility[0] = right_heel.visibility
                visibility[1] = right_ankle.visibility
                visibility[2] = right_index.visibility
                visibility[3] = left_heel.visibility
                visibility[4] = left_ankle.visibility
                visibility[5] = left_index.visibility

                # Calculate the distance between the current position and the start position
                sub_pos = curr_pos - start_pos # 6x3
                sq_pos = np.square(sub_pos[:,:2]) # 6x2
                sum_pos = np.sum(sq_pos, axis=1) # 6x1
                dis_pos = (np.sqrt(sum_pos)).reshape((6,1)) # 6x1

                time_elapsed = time.time() - start_time
                print("Time Left: ", 10 - time_elapsed)

                # Print threshold
                # print("Threshold: ", threshold)

                # print distance
                # print("Distance: ", dis_pos)

                # print("Visibility: ", visibility)
                
                # If the distance is greater than 5 cm then the user lost their balance
                # if np.any(dis_pos > threshold):
                #     print(dis_pos > threshold)
                #     print("Lost Balance")
                #     break

                # if abs(right_ankle[1] - left_ankle[1])
                # print("right ankle = ", right_ankle_pos[1], "left ankle = ", left_ankle_pos[1], "difference = ", abs(right_ankle_pos[1] - left_ankle_pos[1]))
                if (abs(right_ankle_pos[1] - left_ankle_pos[1]) < 15):
                    print("Lost Balance", count)
                    break
            
                # # If the time elapsed is greater than 10 seconds then the user past the test
                if (time_elapsed > 10):
                    print("Passed")
                    break
        
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
