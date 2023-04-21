import cv2
import mediapipe as mp
import numpy as np
import rom
import time
import matplotlib
import scipy
# from skeletonPlot import plot3dClass
# from filter import measure, dynamics
from filterpy import kalman
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints

################################################################################################
# UKF Functions
################################################################################################
def measure(x,numDims):
    # If x is just the positions
    x = x[0:numDims*2]
    return x

def dynamics(x, dt,numDims):
    # This function takes the state and adds the velocity*dt to the position

    tmp = x[0:numDims] + x[numDims:] * dt
    tmp = np.vstack((tmp.reshape((numDims,1)), x[numDims:].reshape((numDims,1))))
    return tmp.flatten()
################################################################################################

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For webcam input:
cap = cv2.VideoCapture(0)
frameTime = time.time()
matplotlib.interactive(True)

# p = plot3dClass()
imageContainer = []
# UKF setup




######################################
# UKF Setup
######################################
imageHeight = 480
imageWidth = 640
#**********************************************************************************************************************
# What do we want to track?
track = 'shoulders'         # Either shoulders or all, though I only have plotting for shoulders right now
velMethod = 'opticalFlow'   # Either opticalFlow or kinematic
plotTogether=True
#**********************************************************************************************************************

if track.lower()=='all':
    numDims = 33*3
elif track.lower()=='shoulders':
    numDims = 2*3


# Initialization of the UKF
points = MerweScaledSigmaPoints(numDims*2, alpha=.1, beta=2., kappa=-1,sqrt_method=scipy.linalg.sqrtm)
# points = JulierSigmaPoints(numDims*2, kappa=-1,sqrt_method=scipy.linalg.sqrtm)
filter = kalman.UnscentedKalmanFilter(numDims*2,numDims,1./20.,measure,dynamics,points)
filter.x = np.zeros(numDims*2)  # Initial State, though this is overwritten by the first measurement
filter.P *= 0.5     # Covariance
filter.R = np.diag(np.ones((numDims*2,))*1) #Measurement Noise
filter.Q = filter.Q*0.1   #Dynamics Noise (Increasing this increases the speed of the update)                                       #Q_discrete_white_noise(dim=2,dt = 1./20.,var = 0.01**2, block_size=numDims,order_by_dim=False)
frameCount = -1
areaSize = 10  # This is 1/2 the size of the area used for optical flow
def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(
        -((x - mx) ** 2. / (2. * sx ** 2.) + (y - my) ** 2. / (2. * sy ** 2.)))


x = np.linspace(-1, 1, areaSize*2)
y = np.linspace(-1, 1, areaSize*2)
x, y = np.meshgrid(x, y)

gauss = gaus2d(x,y)



#######################################################################################################################

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=False) as pose:
    while cap.isOpened():
        frameCount += 1
        success, image = cap.read()
        prevFrameTime = frameTime
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        # image = cv2.GaussianBlur(image, (5, 5), 0)    # Uncommenting this line seems to have a positive effect when smoothing isn't on
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        results = pose.process(image)

        ###########################################
        # UKF - Grab the positions of each joint
        ###########################################
        if results.pose_landmarks is not None:
            if track=='all':
                xyz = [[k.x,k.y,k.z] for k in results.pose_landmarks.landmark]
            elif track=='shoulders':
                xyz = [results.pose_landmarks.landmark[11].x*imageWidth, results.pose_landmarks.landmark[11].y*imageHeight,results.pose_landmarks.landmark[11].z*imageWidth,
                       results.pose_landmarks.landmark[12].x*imageWidth, results.pose_landmarks.landmark[12].y*imageHeight,results.pose_landmarks.landmark[12].z*imageWidth]
            xyz = np.array(xyz).reshape((numDims))
        ############################################

        if 'landmark' in dir(results.pose_world_landmarks):
            #################################################################
            # Need to run the UKF
            #################################################################
            delt = time.time() - prevFrameTime
            if frameCount==0:   # Set the initial state of the filter
                if velMethod=='opticalFlow':
                    prevImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

                state = np.hstack((xyz, np.zeros((numDims)))).flatten()
                filter.x = state
            else:
                # Try a different way to get Velocity:
                if velMethod=='opticalFlow':
                    # In order to get the velocity, I plan on using dense optical flow in an area around the shoulder joints
                    # I will then take the average of the optical flow vectors in that area and use that as the velocity

                    try:
                        xy1 = xyz[0:2].astype(int) # X,Y of the left shoulder
                        xy2 = xyz[3:5].astype(int) # X,Y of the right shoulder
                        curImage = image.copy()
                        curImage = cv2.cvtColor(curImage,cv2.COLOR_BGR2GRAY)
                        lShoulderVel = cv2.calcOpticalFlowFarneback(prevImage[xy1[1]-areaSize:xy1[1]+areaSize,xy1[0]-areaSize:xy1[0]+areaSize],
                                                                    curImage[xy1[1]-areaSize:xy1[1]+areaSize,xy1[0]-areaSize:xy1[0]+areaSize],
                                                                    None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        rShoulderVel = cv2.calcOpticalFlowFarneback(
                            prevImage[xy1[1] - areaSize:xy1[1] + areaSize, xy1[0] - areaSize:xy1[0] + areaSize],
                            curImage[xy1[1] - areaSize:xy1[1] + areaSize, xy1[0] - areaSize:xy1[0] + areaSize],
                            None, 0.5, 3, 15, 3, 5, 1.2, 0)

                        newVel = np.array([np.mean(gauss*lShoulderVel[:,:,0]),
                                  np.mean(gauss*lShoulderVel[:,:,1]),
                                  0.,
                                  np.mean(gauss*rShoulderVel[:,:,0]),
                                  np.mean(gauss*rShoulderVel[:,:,1]),
                                  0.]).flatten()

                        state = np.hstack((xyz, newVel)).flatten()
                    except:
                        vel = (xyz - state[0:numDims]) / delt
                        state = np.hstack((xyz, vel)).flatten()



                    # cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                else:
                    vel = (xyz - state[0:numDims])/delt
                    state = np.hstack((xyz, vel)).flatten()

            filter.predict(dt=delt,numDims=numDims)
            filter.update(state,numDims=numDims)
            #################################################################
            # right_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            # # right_wrist_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
            # right_wrist_pos = np.array([right_wrist.x, right_wrist.y])
            #
            # right_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            # # right_elbow_pos = np.array([right_elbow.x, right_elbow.y, right_elbow.z])
            # right_elbow_pos = np.array([right_elbow.x, right_elbow.y])
            #
            # right_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            # # right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
            # right_shoulder_pos = np.array([right_shoulder.x, right_shoulder.y])
            #
            # rom_right_elbow = rom.rom_right_elbow(right_wrist_pos, right_elbow_pos, right_shoulder_pos)
            # print(rom_right_elbow)
            # print(f'R Upper Arm Length: {rom.limbLength(right_shoulder_pos, right_elbow_pos)}')

        # Draw the pose annotation on the image.
        image.flags.writeable = True



        frameTime = time.time()
        fps = 1/(frameTime-prevFrameTime)

        ##############################################################################################################
        # Want to have a comparison of UKF and raw, so lets stack two images together
        ##############################################################################################################

        if plotTogether == False:
            image2 = image.copy()

            image2 = cv2.line(image2, filter.x[0:2].astype(int), filter.x[3:5].astype(int), color=(0, 0, 255),
                              thickness=2)
            image2 = cv2.circle(image2, filter.x[0:2].astype(int), 5, color=(255, 0, 0), thickness=-1)
            image2 = cv2.circle(image2, filter.x[3:5].astype(int), 5, color=(255, 0, 0), thickness=-1)

            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            image2 = cv2.flip(image2, 1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.flip(image, 1)
            cv2.putText(image, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', np.hstack((image, image2)))

        else:
            image = cv2.line(image, filter.x[0:2].astype(int), filter.x[3:5].astype(int), color=(0, 0, 255),
                              thickness=2)
            image = cv2.circle(image, filter.x[0:2].astype(int), 5, color=(255, 0, 0), thickness=-1)
            image = cv2.circle(image, filter.x[3:5].astype(int), 5, color=(255, 0, 0), thickness=-1)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            image = cv2.flip(image, 1)
            cv2.putText(image, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(image, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('MediaPipe Pose', image)
        ##############################################################################################################

        # Flip the image horizontally for a selfie-view display.

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
