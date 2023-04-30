import cv2
import mediapipe as mp
import numpy as np
import time

def test2():
    # start the timer
    start_time = time.time()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
  
        # initialize variables
        start_y = 0
        frame = 0
        count = 0 # Count how many times the user can stand up and sit down in 30 seconds
        state = 0 # 0 for sitting, 1 for standing
        last_state = 0 # last state

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
                curr_y = right_shoulder_pos[1]

                if (frame == 1):
                    start_y = curr_y

                if abs(curr_y - start_y) < 200:
                    # sitting
                    state = 0
                    if (last_state == 1):
                        count += 1
                        print("count = ", count)
                else:
                    # standing
                    state = 1

                last_state = state        

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            frame += 1
            curr_time = time.time()
            if (curr_time - start_time > 30):
                cap.release()
                return count

            # Flip the image horizontally for a selfie-view display.
            cv2.putText(image, "time = " + str(int(time.time() - start_time)) + " count = " + str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()

def test2_pass(age, gender, count):
    if (age <= 64):
        if (gender == 'male'):
            if (count < 14):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 12):
                return False
            else:
                return True
    elif (age <= 69):
        if (gender == 'male'):
            if (count < 12):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 11):
                return False
            else:
                return True
    elif (age <= 74):
        if (gender == 'male'):
            if (count < 12):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 10):
                return False
            else:
                return True
    elif (age <= 79):
        if (gender == 'male'):
            if (count < 11):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 10):
                return False
            else:
                return True
    elif (age <= 84):
        if (gender == 'male'):
            if (count < 10):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 9):
                return False
            else:
                return True
    elif (age <= 89):
        if (gender == 'male'):
            if (count < 8):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 8):
                return False
            else:
                return True
    else:
        if (gender == 'male'):
            if (count < 7):
                return False
            else:
                return True
        elif (gender == 'female'):
            if (count < 4):
                return False
            else:
                return True