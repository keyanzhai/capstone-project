import cv2
import numpy as np

# Create a VideoCapture object to capture the video stream from the default webcam
cap = cv2.VideoCapture(0)
image2 = cv2.imread('1-1.jpg')

# Loop until the user presses the 'q' key
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    image2 = cv2.resize(image2, (frame.shape[1], frame.shape[0]))

    # Display the frame in a window
    cv2.imshow('Webcam', np.hstack((frame, image2)))

    # Wait for 1 millisecond and check if the user pressed the 'enters' key
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
