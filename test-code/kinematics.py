import cv2
import mediapipe as mp
import numpy as np


testSet = 'v1'
fileName = './test-data/' + testSet + '.mov'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(fileName)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  
  count = 1
  l1List = []
  l2List = []
  kinList = []
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Video ended.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image_height, image_width, _ = image.shape
    results = pose.process(image)

    rS = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    rK = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    rH = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

    # Right shoulder, knee, and hip positions
    rsPos = np.array([rS.x * image_width, rS.y  * image_height])
    rkPos = np.array([rK.x * image_width, rK.y  * image_height])
    rhPos = np.array([rH.x * image_width, rH.y  * image_height])

    #Get link between shoulder and hip
    shPos = rhPos - rsPos
    link2 = shPos/np.linalg.norm(shPos)
    

    #Get link between hip and knee
    hkPos = rkPos - rhPos
    link1 = hkPos/np.linalg.norm(hkPos)
    
    # Moving link lenght window
    l1List.append(np.linalg.norm(hkPos))
    l2List.append(np.linalg.norm(shPos))
    if count > 10:
      l1List.pop(0)
      l2List.pop(0)
    l1Numpy = np.array(l1List)
    l2Numpy = np.array(l2List)
    l1 = np.mean(l1Numpy)
    l2 = np.mean(l2Numpy)

    yAxis = np.array([0, 1])

    #Get angle between link1 and yAxis
    angle1 = np.arccos(np.dot(link1, yAxis))
    dega1 = np.rad2deg(angle1)

    #Get angle between link2 and link1
    angle2 = np.arccos(np.dot(link1,link2))
    dega2 = np.rad2deg(angle2)


    # Based on link1 and angle1, determine user hip position
    if rkPos[0] > rhPos[0]:
      hipPos = np.array([-l1* np.sin(angle1)+rkPos[0], -l1* np.cos(angle1)+rkPos[1]])
    else:
      hipPos = np.array([l1* np.sin(angle1)+rkPos[0], -l1* np.cos(angle1)+rkPos[1]])
    if (rsPos[0] > rhPos[0] and rhPos[0] > rkPos[0]) or (rsPos[0] < rhPos[0] and rkPos[0] < rhPos[0]):
      shoPos = np.array([-l2* np.sin(angle2-angle1)+hipPos[0], -l2* np.cos(angle2-angle1)+hipPos[1]])
    else:
      shoPos = np.array([l2* np.sin(angle2-angle1)+hipPos[0], -l2* np.cos(angle2-angle1)+hipPos[1]])

    kinList.append(shoPos)
    # Draw the circle for the shoulder, hip, and knee
    cv2.circle(image, (int(shoPos[0]), int(shoPos[1])), 5, (255, 0, 0), -1)
    cv2.circle(image, (int(hipPos[0]), int(hipPos[1])), 5, (0, 100, 255), -1)
    cv2.circle(image, (int(rsPos[0]), int(rsPos[1])), 5, (0, 0, 255), -1)
    cv2.circle(image, (int(rhPos[0]), int(rhPos[1])), 5, (0, 0, 255), -1)
    cv2.circle(image, (int(rkPos[0]), int(rkPos[1])), 5, (0, 0, 255), -1)

    count = count + 1

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

kinPos = np.array(kinList)
np.save('./test-data/stickerPos/' + testSet + '-kinPos.npy', kinPos)
cap.release()
