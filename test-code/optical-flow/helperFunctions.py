import numpy as np
import cv2 as cv

class skeleton:
    def __init__(self,p0,numLimbPoints=10,fps=30):
        self.jointPos = p0
        self.numLimbPoints = numLimbPoints
        # Create a dictionary to make joint indexing easier
        self.joints = {
            'head':         0,
            'neck':         1,
            'waist':        2,
            'rShoulder':    3,
            'rElbow':       4,
            'rWrist':       5,
            'rHand':        6,
            'lShoulder':    7,
            'lElbow':       8,
            'lWrist':       9,
            'lHand':        10,
        }
        self.limbPoints = self.generateLimbPoints(self.numLimbPoints)
        self.limbPointVels=dict()
        self.dt = 1./fps
        self.testWristPos = self.jointPos[self.joints['rWrist']]

        self.getJointAngDist()

    def setJointPos(self,pNew):
        # Function is used to set joint positions to new position
        self.p0 = pNew


    def getJointAngDist(self):
        # This function will calculate the joint angles
        self.jointAngles = np.zeros((10,))
        self.jointDists = np.zeros((10,))

        tmp = np.zeros((10,2))

        # First, lets get the X,Y differences
        # Neck to Head
        tmp[0,:] = self.jointPos[0,0,:]-self.jointPos[1,0,:]
        # Waist to Neck
        tmp[1, :] = self.jointPos[1, 0, :] - self.jointPos[2, 0, :]
        # Neck to Right Shoulder
        tmp[2, :] = self.jointPos[3, 0, :] - self.jointPos[1, 0, :]
        # Neck to Left Shoulder
        tmp[3, :] = self.jointPos[7, 0, :] - self.jointPos[1, 0, :]
        # Right Shoulder to Right Elbow
        tmp[4, :] = self.jointPos[4, 0, :] - self.jointPos[3, 0, :]
        # Right Elbow to Right Wrist
        tmp[5, :] = self.jointPos[5, 0, :] - self.jointPos[4, 0, :]
        # Right Wrist to Right Hand
        tmp[6, :] = self.jointPos[6, 0, :] - self.jointPos[5, 0, :]
        # Left Shoulder to Left Elbow
        tmp[7, :] = self.jointPos[8, 0, :] - self.jointPos[7, 0, :]
        # Left Elbow to Left Wrist
        tmp[8, :] = self.jointPos[9, 0, :] - self.jointPos[8, 0, :]
        # Left Wrist to Left Hand
        tmp[9, :] = self.jointPos[10, 0, :] - self.jointPos[9, 0, :]

        # Now, we can get the joint distances and joint angles
        self.jointDists = np.linalg.norm(tmp,axis=1)
        self.jointAngles = np.arctan2(tmp[:,0],tmp[:,1])

        return self.jointDists, self.jointAngles

    def createSkeleton(self,img,p0=[]):
        # First, head connects to center of shoulders
        if not np.any(p0):
            p0 = self.jointPos

        img = cv.line(img,p0[0,0,:].astype(int),p0[1,0,:].astype(int),color=(0, 0, 255),thickness=2)

        # Next, Center of shoulders to left and right shoulder
        img = cv.line(img, p0[1, 0, :].astype(int), p0[3, 0, :].astype(int), color=(0, 0, 255), thickness=2)
        img = cv.line(img, p0[1, 0, :].astype(int), p0[7, 0, :].astype(int), color=(0, 0, 255), thickness=2)

        # Next, shoulders to elbows
        img = cv.line(img, p0[3, 0, :].astype(int), p0[4, 0, :].astype(int), color=(0, 0, 255), thickness=2)
        img = cv.line(img, p0[7, 0, :].astype(int), p0[8, 0, :].astype(int), color=(0, 0, 255), thickness=2)

        # Next, elbows to wrist
        img = cv.line(img, p0[4, 0, :].astype(int), p0[5, 0, :].astype(int), color=(0, 0, 255), thickness=2)
        img = cv.line(img, p0[8, 0, :].astype(int), p0[9, 0, :].astype(int), color=(0, 0, 255), thickness=2)

        # Next, wrist to center of hand
        img = cv.line(img, p0[5, 0, :].astype(int), p0[6, 0, :].astype(int), color=(0, 0, 255), thickness=2)
        img = cv.line(img, p0[9, 0, :].astype(int), p0[10, 0, :].astype(int), color=(0, 0, 255), thickness=2)

        # Finally, center of shoulders to waist
        img = cv.line(img, p0[1, 0, :].astype(int), p0[2, 0, :].astype(int), color=(0, 0, 255), thickness=2)

        # Now lets put a blue dot at every joint
        for row in p0:
            img = cv.circle(img, row[0, :].astype(int), 5, color=(255, 0, 0), thickness=-1)

        return img

    def generateLimbPoints(self, pointsPerLimb=3):
        '''
        One idea is to generate points along the limb between joints to use as comparison. This should allow us to
        hopefully deal with occlusions by checking that the velocities of the limb points agree with the velocity of the
        joint points
        :param pointsPerLimb:
        :return limbPoints: This function will output a dictionary containing points for each limb
        '''

        limbPoints = dict()
        # For debug, lets just focus on the right lower arm

        # First, calculate the distance between the x and y coordinates of the joints

        # Right Lower Arm
        dist = self.jointPos[self.joints['rWrist']]-self.jointPos[self.joints['rElbow']]
        divisions = np.linspace(0,1,pointsPerLimb+1,endpoint=False)[1:]
        limbPoints['lowerRightArm'] = ((dist.T*divisions.reshape((1,-1))).T + self.jointPos[self.joints['rElbow']]).reshape((-1,1,2))

        # Right Upper Arm
        dist = self.jointPos[self.joints['rElbow']]-self.jointPos[self.joints['rShoulder']]
        divisions = np.linspace(0,1,pointsPerLimb+1,endpoint=False)[1:]
        limbPoints['upperRightArm'] = ((dist.T*divisions.reshape((1,-1))).T + self.jointPos[self.joints['rShoulder']]).reshape((-1,1,2))



        return limbPoints


    def getAllTrackingPoints(self):
        # This function returns all tracked points in a stacked numpy array

        tmp = np.vstack((self.jointPos,self.limbPoints['lowerRightArm'],self.limbPoints['upperRightArm'],self.testWristPos.reshape((1,1,2))))

        return tmp.astype(np.float32)

    def setAllTrackingPoints(self,newPoints):
        # This takes the new points and breaks them up into joint and limb points


        self.limbPointVels['lowerRightArm'] = (self.limbPoints['lowerRightArm'] - newPoints[11:21]).reshape((-1,2))

        self.calcJointMotion(newPoints)
        self.jointPos = newPoints[0:11]
        self.limbPoints = self.generateLimbPoints(self.numLimbPoints)  # newPoints[11:]

    def getAngularVelocity(self,jointPos,pointPos,pointVel):
        # This function will generate the angular velocity of all points

        r = pointPos.reshape((-1,2))-jointPos.reshape((-1,2))
        angularVel = np.cross(r,pointVel)/np.linalg.norm(r)**2

        return angularVel

    def calcJointMotion(self,newPoints):

        angVel = self.getAngularVelocity(newPoints[-1],newPoints[11:21],self.limbPointVels['lowerRightArm'])

        avgAngVel = np.mean(angVel)

        r = self.testWristPos-self.jointPos[self.joints['rElbow']]

        phi = np.arctan2(r[:,1],r[:,0])
        mag = np.linalg.norm(r,axis=1)*avgAngVel
        v = np.hstack((mag*np.cos(np.pi/2-phi),mag*np.sin(np.pi/2-phi)))

        self.testWristPos = self.testWristPos + v*self.dt

    def plotTestWristPoint(self,img):

        img = cv.circle(img, self.testWristPos[0].astype(int), 5, color=(0, 255, 0), thickness=-1)

        return img











