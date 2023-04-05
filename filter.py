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
	def __init__(s,Q=0.1,R=0.1,pos=np.zeros((33,1)),vel=np.zeros((33,1))):
		# Q is the covariance matrix for the process noise
		# R is the covariance matrix for the measurement noise
		# pos is the initial position of each joint with size (33,1)
		# vel is the initial velocity of each joint with size (33,1)
		s.Q = Q*np.eye(33)
		s.R = R*np.eye(33)
		s.prevPos = pos
		s.vel = vel
		s.stateCov = np.eye(66)

	def generateSigmaPoints(s,dt=(1.0/20.0)):
		# This function is used to generate a list of sigma points
		sigPoints = 0
		return sigPoints

	def propDynamics(s,dt):

		return 0

	def calcVariance(s):


		return 0

	def calcNextState(s):

		return 0

	def performUKF(s):

		return 0