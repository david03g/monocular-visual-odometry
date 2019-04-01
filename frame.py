import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from helpers import*


class Frame():

	id = 0

	def __init__(self, img, prev):

		Frame.id += 1

		#image for display purposes
		self.img = img

		#preprocessed image for analysis
		self.processedImg = preprocessImg(self.img, 1.23)

		self.kps, self.des = self.extractFeatures()
		self.prev = prev
		self.ptMatches = self.matchFrames()
		self.id = Frame.id

	def extractFeatures(self):

		orb = cv2.ORB.create()

		#extract features from processed image 
		features = [f.ravel() for f in np.int0(cv2.goodFeaturesToTrack(self.processedImg, 3000, qualityLevel = .01, minDistance = 3))]
		kps = [cv2.KeyPoint(x=p[0], y =p[1], _size = 20) for p in features]
		
		#return keypoints and descriptors
		return orb.compute(self.processedImg, kps)

	def annotate(self):

		if self.prev is not None:
			for p0, p1 in self.matchFrames():

				u0, v0 = map(lambda x: int(round(x)), p0)
				u1, v1 = map(lambda x: int(round(x)), p1)
				if u0 == u1 and v0 == v1:
					cv2.circle(self.img, (u1, v1), color=(100, 100, 100), radius = 3)
				else:
					cv2.circle(self.img, (u1, v1), color=(0, 255, 0), radius = 3)
					cv2.circle(self.img, (u0, v0), color=(255, 0, 0), radius = 3)
					cv2.line(self.img, (u0, v0), (u1, v1), color = (255,0,0))

		cv2.putText(self.img, "Speed {} m/s".format(self.getCurrentSpeed()), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255))


	def processFrame(self):

		self.annotate()

		cv2.imshow('Monocular Visual Odometry', self.img)


	def matchFrames(self):

		if self.prev is not None:
			m = cv2.BFMatcher(cv2.NORM_HAMMING)
			matches = m.match(self.des, self.prev.des)

			pts0 = np.float32([self.kps[m.queryIdx].pt for m in matches]).reshape(-1,2)
			pts1 = np.float32([self.prev.kps[m.trainIdx].pt for m in matches]).reshape(-1,2)


			H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 1,  maxIters = 100, confidence = .99999)

			ret = np.array([(pts0[i], pts1[i]) for i in mask.nonzero()[0]])

			#model, inliers = ransac((ret[:, 0], ret[:, 1]), EssentialMatrixTransform, min_samples=8, residual_threshold=.1, max_trials=10)

			return ret

	def getCurrentSpeed(self):
        #TODO
		return 9


	

	'''
	def calcFundamentalMatrix(self):

		if self.prev is not None:

			pts0 = np.array([x[0] for x in self.matches])
			pts1 = np.array([x[1] for x in self.matches])

			fundamental_matrix, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC)

			if fundamental_matrix is None or fundamental_matrix.shape == (1, 1):
				raise Exception('No fundamental matrix found')
			elif fundamental_matrix.shape[0] > 3:
				# more than one matrix found, just pick the first
				fundamental_matrix = fundamental_matrix[0:3, 0:3]
			return np.matrix(fundamental_matrix) 

	'''

		
