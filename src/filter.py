import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


# Unscented Kalman Filter
'''
The filter file is meant to house all classes and functions used to clean up the pose data. I will first start with a
very simple UKF, which will use the previous positions of the joints to calculate velocities, which will be used to
propogate the dynamics
'''

class poseFilter:
	def __init__(s,Q=0.1,R=0.1,pos=np.zeros((33,1)),vel=np.zeros((33,1)),stateSize=66):
		# Q is the covariance matrix for the process noise
		# R is the covariance matrix for the measurement noise
		# pos is the initial position of each joint with size (33,1)
		# vel is the initial velocity of each joint with size (33,1)
		s.size = stateSize
		s.Q = Q*np.eye(s.size)
		s.R = R*np.eye(s.size)
		s.prevPos = pos
		s.vel = vel
		s.stateCov = np.eye(s.size)

	def generateSigmaPoints(s,state,stateCov,dt=(1.0/20.0)):
		'''
		This function is used to generate a list of sigma points
		:param state: This is the current state of the system, and includes both the position and velocity of each joint
		:param stateCov: This is the covariance matrix of the state
		:param dt: Time between frames
		:return: sigPoints: This is a list of sigma points, which will be used to propogate the dynamics
		'''
		# This function is used to generate a list of sigma points
		sigPoints = 0

		# Add the dynamics noise to the state covariance
		nStateCov = stateCov + s.Q*dt

		S = la.sqrtm(nStateCov)
		S = S * np.sqrt(state.shape[0])
		S = np.real(S)

		tmp = np.hstack((S,-S))
		sigPoints = tmp+state

		return sigPoints

	def getState(s,currentPos):
		'''
		This function is used to get the current state of the system
		:param currentPos: This is the current position of the joints
		:return: state: This is the current state of the system
		'''
		state = np.zeros((s.size,1))
		state[0:int(s.size/2),0] = currentPos[:,0]
		s.vel = currentPos - s.prevPos
		state[int(s.size/2):s.size, 0] = s.vel[:, 0]
		return state

	def propDynamics(s,dt):









		return 0

	def calcVariance(s):


		return 0

	def calcNextState(s):

		return 0

	def performUKF(s):

		return 0

if __name__=='__main__':
	pose = poseFilter()
	pose.getState(np.ones((33,1)))
	pose.generateSigmaPoints(np.zeros((66,1)),np.eye(66))