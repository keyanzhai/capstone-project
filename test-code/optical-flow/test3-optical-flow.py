import cv2
import mediapipe as mp
import numpy as np
import time

start_time = time.time()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    start_pos = np.zeros((4,2));
    prev_pos = np.zeros((4,2));
    curr_pos = np.zeros((4,2));
    frame_cnt = 0
    
    while cap.isOpened():
        success, image = cap.read() # Get current image
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        
        # Get frame image in grayscale
        frame_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
        # Update frame count
        frame_cnt += 1

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        print("frame_cnt = ", frame_cnt)

        # Get positions for the 6 feet joints using mediapipe as the initial positions
        results = pose.process(image)
        if 'landmark' in dir(results.pose_world_landmarks):
            # Get the (x, y) pixel position of the 6 feet joints of the current frame
            right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]
            left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
            right_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            left_index = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]

            right_heel_pos = np.array([right_heel.x * image_width, right_heel.y * image_height])
            left_heel_pos = np.array([left_heel.x * image_width, left_heel.y * image_height])
            right_index_pos = np.array([right_index.x * image_width, right_index.y * image_height])
            left_index_pos = np.array([left_index.x * image_width, left_index.y * image_height])
        elif (frame_cnt == 30):
            print("No feet joints detected...")
            break

        if (frame_cnt < 30):
            print("frame_cnt < 30, skip...")
        elif (frame_cnt == 30):
            print("frame_cnt = 30, set init positions...")
            start_time = time.time() # start the timer

            # Positions of the 6 joints of the current frame
            # Size: (4, 2)
            curr_pos = np.array([right_heel_pos, right_index_pos, left_heel_pos, left_index_pos])
            right_foot_length = np.linalg.norm(right_heel_pos - right_index_pos)
            print("right_foot_length = ", right_foot_length)

            start_pos = curr_pos # (4, 2)
            prev_pos = curr_pos # (4, 2)
            frame_old_img = frame_img.copy()
            print("initial positions: ", start_pos)
        else:
            print("frame_cnt > 30, start optical flow...")
            # Calculate the optical flow
            prev_pos = prev_pos.reshape(-1,1,2).astype(np.float32)
            curr_pos, st, err = cv2.calcOpticalFlowPyrLK(frame_old_img, frame_img, prev_pos, None, **lk_params)
            curr_pos = curr_pos.reshape(-1,2)
            print("curr_pos = ", curr_pos)

            # Calculate the distance between the current position and the start position
            sub_pos = curr_pos - start_pos # 6x2
            sq_pos = np.square(sub_pos) # 6x2
            sum_pos = np.sum(sq_pos, axis=1) # 6x1
            dis_pos = (np.sqrt(sum_pos)).reshape((4,1)) # 6x1
            threshold = 0.25 * right_foot_length * np.ones((4,1))

            print("dis_pos = ", dis_pos)

            # If there's a movement of the feet, the test fails
            if np.any(dis_pos > threshold):
                time_elapsed = time.time() - start_time
                print("Failed")
                print("time_elapsed = ", time_elapsed)
                break

            # Update the previous position
            prev_pos = curr_pos
            # Update the previous frame
            frame_old_img = frame_img.copy()

        # If the time elapsed is greater than 10 seconds then the user past the test
        time_elapsed = time.time() - start_time
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

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
cap.release()
