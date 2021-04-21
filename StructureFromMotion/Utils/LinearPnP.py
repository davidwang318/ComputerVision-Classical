import numpy as np
import cv2
import matplotlib.pyplot as plt
from GetInlierRANSANC import * 
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *


def linearPnP(worldPts, imgPts, K):
	# Construct the homogeneous 3d pts
    homoAxis = np.ones((len(worldPts), 1))
    worldPts_homo = np.concatenate((worldPts, homoAxis), axis=1)
    imgPts_homo = np.concatenate((imgPts, homoAxis), axis=1)
    # Normalize the img points
    imgPts_normal = np.matmul(imgPts, np.linalg.inv(K).T)
    imgPts_normal /= imgPts_normal[:,2][..., None]
    # Initialize the pnp matrix
    A = None
    # Construct the A matrix
    for world_pt, img_pt in zip(worldPts_homo, imgPts_normal):
    	u, v = img_pt[:2]
    	A_tmp = np.array([[0,0,0,0, -world_pt[0],-world_pt[1],-world_pt[2],-world_pt[3] , v*world_pt[0],v*world_pt[1],v*world_pt[2],v*world_pt[3]],
    		              [world_pt[0],world_pt[1],world_pt[2],world_pt[3], 0,0,0,0, -u*world_pt[0],-u*world_pt[1],-u*world_pt[2],-u*world_pt[3]],
    		              [-v*world_pt[0],-v*world_pt[1],-v*world_pt[2],-v*world_pt[3], u*world_pt[0],u*world_pt[1],u*world_pt[2],u*world_pt[3], 0,0,0,0]])
    	if A is None:
    		A = A_tmp
    	else:
    		A = np.concatenate((A, A_tmp), axis=0)
    # Solving the equation using SVD decomposition
    u, s, vh = np.linalg.svd(A)
    P = vh[-1]
    P = P.reshape((3,4))

    return np.matmul(K, P)


if __name__ == '__main__':
    filePath = "../Data/Imgs/"
    savePath = "../Data/Imgs/GetInlierRANSAC/"
    corDict = readMatches(filePath)
    curDict = corDict['12']

    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    imgL = cv2.imread(filePath+str(1)+'.jpg')
    imgR = cv2.imread(filePath+str(2)+'.jpg')
    F, matchPts_inlier = GetInlierRANSAC(curDict, 50000, 0.0025)
    imgMatch = drawMatching(imgL, imgR, curDict, matchPts_inlier)

    cv2.imshow("Match12", imgMatch)
    # cv2.imwrite(savePath + "Match"+str(i)+str(j) + '.jpg', imgMatch)
    cv2.waitKey()
    cv2.destroyAllWindows()

    E = EssentialMatrixFromFundamentalMatrix(F, K)
    poses = ExtractCameraPose(E)
    best_pose, best_point, best_inlier = DisambiguateCameraPose(poses, matchPts_inlier, K)
    # best_point: 3d world point(n*3), best_inlier: 2d homo image point(n*2*3)
    PlotTriangulation([best_pose], [best_point])

    # Use the best_inlier to calculate the pose of camera 2:
    P_pnp = linearPnP(best_point, best_inlier[:, 1, :2], K)
    print(P_pnp)

