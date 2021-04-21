import numpy as np
import cv2
from GetInlierRANSANC import * 
from GetIntrinsic import *


def EssentialMatrixFromFundamentalMatrix(F, K):
    print("*** Essential matrix process:")
    E = np.matmul(np.matmul(K.T, F), K)
    u, s, vh = np.linalg.svd(E)
    newValues = np.ones(s.shape)
    newValues[-1] = 0
    reconstE = np.matmul(u, (newValues[..., None]*vh))
    print(reconstE)
    print("==========")
    return reconstE

if __name__ == "__main__":
    # img1 = cv2.imread('frame_gray/frame100.jpg') # queryImage
    # img2 = cv2.imread('frame_gray/frame101.jpg') # trainImage
    # pt = GetSiftCorrespondence(img1, img2, 0.7)
    # F, _ = GetInlierRANSAC(pt, 5000, 0.001)
    # E = EssentialMatrixFromFundamentalMatrix(F)
    K = np.array([[568.996140852, 0, 643.21055941],
              [0, 568.988362396, 477.982801038],
              [0, 0, 1]])
    filePath = "../Data/Imgs/"
    corDict = readMatches(filePath)
    curDict = corDict['12']
    F, matchPts_inlier = GetInlierRANSAC(curDict, 20000, 0.0025)
    E = EssentialMatrixFromFundamentalMatrix(F, K)

    # print(E)
