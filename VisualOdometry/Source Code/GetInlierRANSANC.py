import sys
import numpy as np
import cv2
import random
from EstimateFundamentalMatrix import *


def GetInlierRANSAC(matchPts, iteration, threshold):
    print("*** RANSAC process:")
    # Initialize necessary parameters
    numPts = matchPts.shape[0]
    inNum, inList = 0, []

    # Create homogeneous pts
    projAxis=  np.ones((numPts, 1))
    pts1 = np.concatenate((matchPts[:, 0, :], projAxis), axis=1)
    pts2 = np.concatenate((matchPts[:, 1, :], projAxis), axis=1)

    # RANSAC procedure starts
    while iteration > 0:
        # Randomly pick 8 points
        indices = np.random.randint(numPts, size=8)
        randPts1 = pts1[indices]
        randPts2 = pts2[indices]
        # Estimate fundamental matrix from the random pich
        tmpF = EstimateFundamentalMatrix(randPts1, randPts2)
        # Count the inliers
        tmpNum, tmpList = countInlier(tmpF, pts1, pts2, threshold)
        if(tmpNum > inNum):
            F_Best, inNum, inList = tmpF, tmpNum, tmpList
        iteration -= 1

    # Organize the data
    inPts1, inPts2 = pts1[inList], pts2[inList]
    F_Best = EstimateFundamentalMatrix(inPts1, inPts2)
    matchPts = np.concatenate((inPts1[:,None,:], inPts2[:,None,:]), axis=1)
    
    print("Number of Inliers: ", inNum)
    print(F_Best)
    print("==========")

    return F_Best, matchPts

def countInlier(F, pts1, pts2, threshold):
    tmp = np.dot(pts1, F.T)
    inList = abs(np.sum(pts2*tmp, axis=1)) < threshold
    numIns = np.count_nonzero(inList)
    return numIns, inList

def drawMatching(img1, img2, matchPts):
    r1, c1 = img1.shape
    r2, c2 = img2.shape
    imgMatch = np.zeros((max(r1, r2), c1+c2)).astype(np.uint8)
    imgMatch[:r1, :c1] = img1[:, :]
    imgMatch[:r2, c1:] = img2[:, :]
    for pt1, pt2 in matchPts.astype(np.int64):
        x1, y1, _ = pt1
        x2, y2, _ = pt2
        x2 = x2 + c1
        imgMatch = cv2.line(imgMatch, (x1,y1), (x2,y2), 0, 2)
    return imgMatch

