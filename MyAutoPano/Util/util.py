#!/usr/bin/evn python

import cv2
import numpy as np
import sys 
import os

sys.dont_write_bytecode = True

def ANMS(points, nBest_):
	nums = points.shape[0]
	r = sys.maxint*np.ones((nums, 1))
	# create a hash that maps each r to each pair of points
	rHash = np.concatenate((r, points), axis=1)
	# points already sorted so second loop start end at i-1
	for i in range(nums):
		for j in range(i):
			rTmp = np.linalg.norm(points[i] - points[j])
			if rHash[i, 0] > rTmp: rHash[i, 0] = rTmp
	# Sort the distance in descending order along with the point
	bestPts_ = rHash[np.argsort(-rHash[:, 0])][:nBest_, 1:]

	return bestPts_.astype(np.int64)

def createDescriptor(img_, featurePts_, patchSize_):
	# Define the descriptor size
	desSize = (12, 12)
	descriptor = []
	# Paddling the image
	r = int(patchSize_/2)
	img_ = np.pad(img_, r, 'constant', constant_values=0)
	# Calulate descriptors
	for pt in featurePts_:
		patchImg = img_[pt[1]:(pt[1]+2*r+1), pt[0]:pt[0]+2*r+1]
		patchImg = cv2.GaussianBlur(patchImg, (3,3), 0)
		subImg = cv2.resize(patchImg, desSize)
		des = stdDes(subImg.reshape((1, -1)))
		descriptor = np.concatenate((descriptor, des), axis=0) if len(descriptor) else des

	return descriptor

def featureMatching(pts1, feat1, pts2, feat2, ratio=1):
	matchPts = []
	for i, f1 in enumerate(feat1):
		best1, best2, bestInd = sys.maxint, sys.maxint, -1
		for j, f2 in enumerate(feat2):
			error = np.linalg.norm((f1 - f2))**2
			if (error < best1):
				best2 = best1
				best1 = error
				bestInd = j
		if (best1/best2 < ratio):
			matchPts.append(np.array([pts1[i], pts2[bestInd]]))
	matchPts = np.array(matchPts)

	return np.array(matchPts)

def drawMatching(img1, img2, matchPts):
	r1, c1, _ = img1.shape
	r2, c2, _ = img2.shape
	imgMatch = np.zeros((max(r1, r2), c1+c2, 3)).astype(np.uint8)
	imgMatch[:r1, :c1, :] = img1[:, :, :]
	imgMatch[:r2, c1:, :] = img2[:, :, :]
	for pt1, pt2 in matchPts:
		x1, y1 = pt1
		x2, y2 = pt2
		x2 = x2 + c1
		imgMatch = cv2.line(imgMatch, (x1,y1), (x2,y2), (0,255,0), 1)
	
	return imgMatch

def homoRANSAC(matchPts, threshold, iteration, ratio=1):
	# Initialize parameters
	inRatio, inNum, inList = 0, 0, []
	matchNum = len(matchPts)
	# Start RANSAC
	while(iteration > 0 and inRatio < ratio):
		# Generate 4 random points
		randInd = np.random.randint(matchNum, size=4)
		randPts = matchPts[randInd].astype(np.float32)
		# Compute homography
		a, b = np.array(randPts[:,0,:]), np.array(randPts[:,1,:])
		tmpH = cv2.getPerspectiveTransform(a,b)
		tmpH = tmpH / tmpH[2,2]
		# Count and update inlier
		tmpNum, tmpRatio, tmpList = countInlier(tmpH, matchPts, threshold)
		if(tmpNum > inNum):
			inNum, inRatio, inList = tmpNum, tmpRatio, tmpList
		iteration -= 1
	print("Number of Inliers: ", inNum)

	# Compute Homography based on the inliers
	inlierPts = matchPts[inList]
	H, inList2 = cv2.findHomography(inlierPts[:,0,:], inlierPts[:,1,:], cv2.RANSAC, 5.0)
	# H, _ = cv2.findHomography(inlierPts[:,0,:], inlierPts[:,1,:])


	return H, matchPts[inList]

def countInlier(H, matchPts, threshold):
	matchNum = len(matchPts)
	pts1, pts2 = matchPts[:,0,:].astype(np.float32), matchPts[:,1,:].astype(np.float32)
	hpts1 = cv2.perspectiveTransform(np.array([pts1]), H)[0]
	diffPts = hpts1 - pts2
	inlierList = np.linalg.norm(diffPts, axis=1) < threshold
	inlier = np.count_nonzero(inlierList)

	return inlier, inlier/float(matchNum), inlierList


def stitchHomo(img, H, corOld):
	# Get the x and y of the warped image
	corNew = cv2.perspectiveTransform(corOld, H)[0]
	xMin, yMin = np.amin(corNew, axis=0)
	xMax, yMax = np.amax(corNew, axis=0)
	xNew = int(xMax - xMin)
	yNew = int(yMax - yMin)
	# Calculate the new homography that offset the original one
	offsetH = np.eye(3)
	offsetH[0,2] = -xMin
	offsetH[1,2] = -yMin
	newH = np.dot(offsetH, H)
	# Warp the image 
	newPts = cv2.perspectiveTransform(corOld, newH)
	imgWarp = cv2.warpPerspective(img, newH, (xNew, yNew))
	cv2.imshow('', imgWarp)
	
	return newH, (xNew, yNew), newPts

def stitchDistance(H, matchPts):
	pts1, pts2 = matchPts[:,0,:].astype(np.float32), matchPts[:,1,:].astype(np.float32)
	hpts1 = cv2.perspectiveTransform(np.array([pts1]), H)[0]
	diffPts = hpts1 - pts2
	return np.mean(diffPts, axis=0)

def blendImg(imgSrc, imgDst, fourPts, H, matchPts, poisson):
	yDst, xDst = imgDst.shape[0:2]
	offsetH, imgSize, newPts = stitchHomo(imgSrc, H, fourPts)

	imgWarp = cv2.warpPerspective(imgSrc, offsetH, imgSize)
	cv2.imwrite("Output/Set1/imgWarp.jpg", imgWarp)
	
	xStitch, yStitch = stitchDistance(offsetH, matchPts)
	tempSize, offsetH2 = np.zeros(2), np.eye(3)	
	if(xStitch >= 0):
		tempSize[0] = max(imgSize[0], xDst+xStitch)
		xStitch2 = int(xStitch)
	else:
		offsetH2[0,2] = abs(xStitch)
		tempSize[0] = max((imgSize[0]+abs(xStitch)), xDst)
		xStitch2 = 0

	if(yStitch >= 0):
		tempSize[1] = max(imgSize[1], yDst+yStitch)
		yStitch2 = int(yStitch)
	else:
		offsetH2[1,2] = abs(yStitch)
		tempSize[1] = max((imgSize[1]+abs(yStitch)), yDst)
		yStitch2 = 0

	stitchH = np.dot(offsetH2, offsetH)
	imgBlend = cv2.warpPerspective(imgSrc, stitchH, tuple(tempSize.astype(np.int64)))

	if(poisson):
		center = (xStitch2 + xDst//2, yStitch2 + yDst//2)
		mask1 = np.all(imgBlend[yStitch2:yStitch2+yDst, xStitch2:xStitch2+xDst, :] == 0, axis=2)
		mask1 = mask1[..., None]
		mask1 = np.concatenate((np.concatenate((mask1, mask1), axis=2), mask1), axis=2)
		np.putmask(imgBlend[yStitch2:yStitch2+yDst, xStitch2:xStitch2+xDst, :], mask1, imgDst)
		mask2 = np.ones(imgDst.shape[:2], np.uint8)*255
		imgBlend =cv2.seamlessClone(imgDst, imgBlend, mask2, center, cv2.NORMAL_CLONE)
	else:
		imgBlend[yStitch2:yStitch2+yDst, xStitch2:xStitch2+xDst, :] = imgDst
		

	return imgBlend, newPts

def showImg(name, img):
	cv2.namedWindow(name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(name, 900, 900)
	cv2.imshow(name, img)



def stdDes(des):
	des = des - np.mean(des)
	return des / np.std(des)
	