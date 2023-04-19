import numpy as np
import cv2 as cv
import argparse
from helperFunctions import skeleton
import imageio
import time

filename = './TestVideos/slow_traffic_small.mp4'

cap = cv.VideoCapture(filename)
vidFPS = cap.get(cv.CAP_PROP_FPS)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# ret, frame = cap.read()
# # In order to input points to track, use the first frame of the image, and select points:
frame=old_frame
trackPoints = []
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        trackPoints.append((x,y))
        # displaying the coordinates
        # on the image window
        # font = cv.FONT_HERSHEY_SIMPLEX
        # cv.putText(frame, str(x) + ',' +
        #             str(y), (x, y), font,
        #             1, (255, 0, 0), 2)
        cv.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        cv.imshow('image', frame)


# cv.imshow('image', frame)
#
# # setting mouse handler for the image
# # and calling the click_event() function
# cv.setMouseCallback('image', click_event)
#
# # wait for a key to be pressed to exit
# cv.waitKey(0)
#
# # close the window
# cv.destroyAllWindows()

# Billiards
# p0 = np.array([(130, 67), (155, 139), (143, 148), (166, 150), (155, 158)],dtype=np.float32).reshape((-1,1,2))

# Fencing
# p0 = np.array([(85, 80), (84, 105), (84, 125), (102, 139), (69, 136), (99, 155), (108, 190), (68, 96), (60, 114), (53, 159), (30, 183)],dtype=np.float32).reshape((-1,1,2))

# Optical Flow Test
p0 = np.array([(1077, 239), (1091, 333), (1072, 663), (1006, 338), (974, 476), (1016, 548), (1032, 560), (1214, 357), (1213, 505), (1117, 564), (1090, 574)],dtype=np.float32).reshape((-1,1,2))

skel = skeleton(p0,fps=vidFPS)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
frameContainer = []
timeContainer = []
frameCount = 0

timeContainer.append(time.time())
hsv = np.zeros_like(frame)
hsv[..., 1] = 255
while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    if frameCount==360:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow (SPARSE)

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, skel.getAllTrackingPoints(), None, **lk_params)#p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        # good_new = p1[st==1]
        # good_old = p0[st==1]
        good_new = p1
        good_old = p0

    frame = skel.createSkeleton(frame,p1)



    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    # skel.setJointPos(p0)
    skel.setAllTrackingPoints(p1)

    frame = skel.plotTestWristPoint(frame)

    img = cv.add(frame, mask)
    frameContainer.append(img)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break



    timeContainer.append(time.time())

    # Calculate Optical Flow (DENSE)

    # flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # bgr = np.vstack((bgr,frame))
    # cv.imshow('frame2', bgr)
    # old_gray = frame_gray.copy()
    # k = cv.waitKey(30) & 0xff
    # if k == 27:
    #     break








    frameCount += 1
    # time.sleep(0.5)

cv.destroyAllWindows()

frameRate = []

for i in range(0,len(timeContainer)-1):
    frameRate.append(1./(timeContainer[i+1]-timeContainer[i]))




print("Saving GIF file")
with imageio.get_writer("testing.gif", mode="I") as writer:
    for idx, frame in enumerate(frameContainer):
        if idx%25:
            print("Adding frame to GIF file: ", idx + 1)

        frame = cv.putText(frame, 'FPS: ' + str(np.round(frameRate[idx],2)), (20, 40), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
        frame = cv.putText(frame, 'Frame #: ' + str(idx), (1000, 40), cv.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        writer.append_data(rgb_frame)


