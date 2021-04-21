import numpy as np
import cv2
from EssentialMatrixFromFundamentalMatrix import *
from GetInlierRANSANC import * 
from GetIntrinsic import *


def ExtractCameraPose(E):
    print("*** ExtractCameraPose process:")
    u, s, vh = np.linalg.svd(E, full_matrices=True)
    print(np.linalg.det(u))
    print(np.linalg.det(vh))
    # print("singular value: ", s)    
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = np.matmul(u, np.matmul(w,vh))
    R2 = np.matmul(u, np.matmul(w.T, vh))
    # print("R1: ", np.linalg.det(R1))
    # print("R2: ", np.linalg.det(R2))
    U = u[:, 2]
    if np.linalg.det(R1) < 0: R1 = -R1
    if np.linalg.det(R2) < 0: R2 = -R2
    pose = [(U, R1), (-U, R1), (U, R2), (-U, R2)]

    for i, p in enumerate(pose):
        print("C"+str(i+1))
        print(p[0])
        print("R"+str(i+1))
        print(p[1])
    print("==========")
    return pose


if __name__ == "__main__":
    # img1 = cv2.imread('frame_gray/frame100.jpg', 0) # queryImage
    # img2 = cv2.imread('frame_gray/frame101.jpg', 0) # trainImage
    # pt = GetSiftCorrespondence(img1, img2, 0.7)
    # F, _ = GetInlierRANSANC(pt, 20, 0.001)
    # K = GetIntrinsic("Oxford_dataset/model")
    # E = EssentialMatrixFromFundamentalMatrix(F, K)
    # pose = ExtractCameraPose(E)
    # print(pose)
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    filePath = "../Data/Imgs/"
    corDict = readMatches(filePath)
    curDict = corDict['12']
    F, matchPts_inlier = GetInlierRANSAC(curDict, 20000, 0.0025)
    E = EssentialMatrixFromFundamentalMatrix(F, K)
    pose = ExtractCameraPose(E)

