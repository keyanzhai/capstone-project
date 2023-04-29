import cv2
import numpy as np

# Define the number of corners in the calibration pattern
pattern_size = (8, 11)

# Define the size of the squares in the calibration pattern (in meters)
square_size = 0.023

# Create arrays to store the object points and image points for calibration
obj_points = []
img_points = []

# Load the calibration images
fileLoc = "test-data/calibration/"
calib_images = []
for i in range(1,25):
    calib_images.append(fileLoc + "c" + str(i) + ".jpg")

i = 0
# Loop over the calibration images
for calib_image in calib_images:
    # Load the image
    img = cv2.imread(calib_image)

    cv2.imshow("img", img)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the calibration pattern
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    print(i)
    print(ret)
    print(corners)
    i += 1
    # If the corners are found, add the object and image points to the lists
    if ret == True:
        obj_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        obj_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        obj_points.append(obj_point)
        img_points.append(corners)
    if cv2.waitKey(100) & 0xFF == ord('q'):
      break

# Calibrate the camera
ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients to a file
np.savetxt('camera_matrix.txt', camera_matrix)
np.savetxt('distortion_coefficients.txt', distortion_coefficients)
