import numpy as np
import cv2

#TODO: Change this to the test set you want to use
testSet = 'v2'
videoPlayback = True # If you want video playback set this to True
videoSpeed = 100 # The higher the number the slower the video

# Filenames
stickerFile = 'test-data/stickerPos/'+testSet+'-stickerPos.npy'
mpFile = 'test-data/stickerPos/'+testSet+'-mpPos.npy'
ukfFile = 'test-data/stickerPos/'+testSet+'-ukfPos.npy'
videoFile = 'test-data/'+testSet+'.mov'

# Load data
stickerPos = np.load(stickerFile)
mpPos = np.load(mpFile)
ukfPos = np.load(ukfFile)

# Print the shape of the data
# print("stickerPos shape:", stickerPos.shape)
# print("mpPos shape:", mpPos.shape)
# print("ukfPos shape:", ukfPos.shape)

# find the rows that contain -1 (no red color found)
rows_with_neg_ones = np.argwhere(np.any(stickerPos == -1, axis=1))[:, 0]

# # remove the rows with -1
# stickerPos = np.delete(stickerPos, rows_with_neg_ones, axis=0)
# mpPos = np.delete(mpPos, rows_with_neg_ones, axis=0)
# ukfPos = np.delete(ukfPos, rows_with_neg_ones, axis=0)

# Calculate the bias from the stickerPos to the mpPos
bias = mpPos[3,:] - stickerPos[3,:]

# Get the true position of where mediapipe thinks the right shoulder is
truePos = stickerPos + bias

# We couldn't find the red dot for some frames so we will set them to zero so that they don't affect the error
# I don't remove it because I want to use the data for video playback
truePos[rows_with_neg_ones,:] = [0,0]
mpPos[rows_with_neg_ones,:] = [0,0]
ukfPos[rows_with_neg_ones,:] = [0,0]

# Calculate the distance between the true position and the mp position
mpError = np.sqrt((mpPos[:,0] - truePos[:,0])**2 + (mpPos[:,1] - truePos[:,1])**2)


# Calculate the distance between the true position and the ukf position
ukfError = np.sqrt((ukfPos[:,0] - truePos[:,0])**2 + (ukfPos[:,1] - truePos[:,1])**2)

print("############ RESULTS "+testSet+ " Test ############")
# Calculate the sum of the error
mpErrorSum = np.sum(mpError)
ukfErrorSum = np.sum(ukfError)
print("--------Total Distance--------")
print("MediaPipe Total Distance Error: ", mpErrorSum)
print("UKF Total Distance Error: ", ukfErrorSum)

# Calculate the average error
mpErrorAvg = np.average(mpError)
ukfErrorAvg = np.average(ukfError)
print("--------Average Distance--------")
print("MediaPipe Average Distance Error: ", mpErrorAvg)
print("UKF Average Distance Error:", ukfErrorAvg)

if videoPlayback:
    # Playback the video with calculated positions
    cap = cv2.VideoCapture(videoFile)
    frame = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # Draw a green circle at the true position
        x = int(truePos[frame,0])
        y = int(truePos[frame,1])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # Draw a red circle at the mp position
        x = int(mpPos[frame,0])
        y = int(mpPos[frame,1])
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

        # Draw a blue circle at the ukf position
        x = int(ukfPos[frame,0])
        y = int(ukfPos[frame,1])
        cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow('MediaPipe vs UKF', image)
        if cv2.waitKey(videoSpeed) & 0xFF == 27:
            break
        frame += 1
    cap.release()