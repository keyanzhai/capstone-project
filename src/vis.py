import cv2
import numpy as np

def vis_directions(test_idx):
    # Create a VideoCapture object to capture the video stream from the default webcam
    cap = cv2.VideoCapture(0)

    # # TODO: add images for each test
    # # TODO: add text directions on the images
    if (test_idx == 1):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 1 - Timed Up & Go (TUG)"
    elif (test_idx == 2):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 2 - 30-Second Chair Stand"
    elif (test_idx == 3):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 3 Stage 1 - Stand with your feet side-by-side"
    elif (test_idx == 4):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 3 Stage 2 - Place the instep of left foot so it is touching the big toe of the right foot"
    elif (test_idx == 5):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 3 Stage 3 - Place left foot in front of the right foot, heel touching toe"
    elif (test_idx == 6):
        image2 = cv2.imread('../test-img/1-1.jpg')
        test_text = "Test 3 Stage 4 - Stand on right foot"

    print("Press 'enter' to continue...")

    # Loop until the user presses the 'q' key
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        # image2 = cv2.resize(image2, (frame.shape[1], frame.shape[0]))

        # Display the frame in a window
        # cv2.imshow('Webcam', np.hstack((frame, image2)))
        cv2.putText(frame, test_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)

        # Wait for 1 millisecond and check if the user pressed the 'enters' key
        if cv2.waitKey(1) == 13:
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()
