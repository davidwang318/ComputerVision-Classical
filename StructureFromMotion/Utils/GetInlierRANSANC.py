import sys
import numpy as np
import cv2
import random
from EstimateFundamentalMatrix import *
from ReadMatches import *


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
    
    print("Number of Points: ", numPts)
    print("Number of Inliers: ", inNum)
    print(F_Best)
    print("==========")

    return F_Best, matchPts

def countInlier(F, pts1, pts2, threshold):
    tmp = np.dot(pts1, F.T)
    inList = abs(np.sum(pts2*tmp, axis=1)) < threshold
    numIns = np.count_nonzero(inList)
    return numIns, inList

def drawMatching(img1, img2, matchPts, matchPts2 = None):
    r1, c1, _ = img1.shape
    r2, c2, _ = img2.shape
    imgMatch = np.zeros((max(r1, r2), c1+c2, 3)).astype(np.uint8)
    imgMatch[:r1, :c1] = img1[:, :]
    imgMatch[:r2, c1:] = img2[:, :]
    for pt1, pt2 in matchPts.astype(np.int64):
        x1, y1 = pt1
        x2, y2 = pt2
        x2 = x2 + c1
        imgMatch = cv2.line(imgMatch, (x1,y1), (x2,y2), (0, 0, 255), 1)
    if matchPts2 is not None:
        for pt1, pt2 in matchPts2.astype(np.int64):
            x1, y1, _ = pt1
            x2, y2, _ = pt2
            x2 = x2 + c1
            imgMatch = cv2.line(imgMatch, (x1,y1), (x2,y2), (0, 255, 0), 1)
    return imgMatch


if __name__ == "__main__":

    filePath = "../Data/Imgs/"
    savePath = "../Data/Imgs/GetInlierRANSAC/"
    corDict = readMatches(filePath)

    for i in range(1, 7):
        for j in range(i+1, 7):
            if(len(corDict[str(i)+str(j)])) == 0:
                continue
            imgL = cv2.imread(filePath+str(i)+'.jpg')
            imgR = cv2.imread(filePath+str(j)+'.jpg')
            curDict = corDict[str(i)+str(j)]
            F, matchPts_inlier = GetInlierRANSAC(curDict, 20000, 0.0025)
            imgMatch = drawMatching(imgL, imgR, curDict, matchPts_inlier)

            cv2.imshow("Match"+str(i)+str(j), imgMatch)
            cv2.imwrite(savePath + "Match"+str(i)+str(j) + '.jpg', imgMatch)

    cv2.waitKey()
    cv2.destroyAllWindows()


    # img1 = cv2.imread(filePath+'1.jpg')
    # img2 = cv2.imread(filePath+'2.jpg')

    # F_12, matchPts_12 = GetInlierRANSAC(corDict['12'], 10000, 0.0025)

    # imgMatch = drawMatching(img1, img2, corDict['12'], matchPts_12)

    # cv2.imshow("img1", img1)
    # cv2.imshow("img2", img2)
    # cv2.imshow("Match", imgMatch)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
