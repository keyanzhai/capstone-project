import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# # Create a VideoCapture object to capture the video stream from the default webcam
cap = cv2.VideoCapture('test-data/v1.mov')
frameNum = 0
noRedFrames = 0
# Red color range in BGR

# Ideal Values for v1, v2, v3
lower_red = np.array([87, 78, 170])
upper_red = np.array([125, 100, 210])

# lower_red = np.array([87, 78, 170])
# upper_red = np.array([130, 115, 212])
stickerPos = []
with mp_pose.Pose(
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  

        # Get human segmentation mask
        mask = results.segmentation_mask
        mask = np.where(mask > 0.9, 1.0, 0.0).astype(np.uint8)
        mask = np.stack((mask, mask, mask), axis=2)
        humanImage = mask * image

        # Create a mask for the red color range
        mask = cv2.inRange(humanImage, lower_red, upper_red)

        # Find coordinates of the red color
        coord = cv2.findNonZero(mask)

        # Average the x and y coordinates
        if coord is None:
            print("No red color found")
            noRedFrames += 1
            stickerPos.append((-1, -1))
            continue
        
        x = coord[:, 0, 0]
        y = coord[:, 0, 1]
        x_avg = int(np.median(x))
        y_avg = int(np.median(y))
        stickerPos.append((x_avg, y_avg))

        # Draw a circle on the image
        cv2.circle(humanImage, (x_avg, y_avg), 3, (0, 0, 255), -1)        

        cv2.imshow('Frame', humanImage)
        
        # Save the frame
        # filename = 'test-data/v1frames/v1-frame'+str(frameNum)+'.jpg'
        # cv2.imwrite(filename, humanImage)
        # frameNum += 1

        if cv2.waitKey(1) & 0xFF == 27:
            break

print("Number of frames with no red color: ", noRedFrames)
cap.release()

# Convert the list of tuples to a numpy array
# stickerPos = np.array(stickerPos)
# np.save('test-data/v1frames/v3-stickerPos.npy', stickerPos)