import numpy as np
import cv2
import matplotlib.pyplot as plt
from GetInlierRANSANC import * 
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *

def DisambiguateCameraPose(poses, corr, K):
    # Define the Intrinsic matrix
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    # Construct the projection matrix of the first frame
    pose_world = (np.zeros(3), np.identity(3))
    proj1 = np.matmul(K, (np.concatenate([pose_world[1], pose_world[0][:, None]], axis=1)))
    # Initilize some parameters
    best_pose = pose_world
    n = 0
    X_all = []
    for pose in poses:
        
        proj2 = np.matmul(K, (np.concatenate([pose[1], pose[0][:, None]], axis=1)))
        
        X = LinearTriangulation(proj1, proj2, corr)[..., :-1]
        X_all.append(X)

        Rz = (pose[1])[-1]

        cheirality = np.matmul(X, (Rz)) + pose[0][2] > 0
        cheirality &= X[:, 2] > 0

        z = X[cheirality, :]
        if z.shape[0] > n:
            n = z.shape[0]
            best_pose = pose
            best_point = z
            best_inlier = corr[cheirality, :, :]

    # PlotTriangulation(poses, X_all)

    return best_pose, best_point, best_inlier

def PlotTriangulation(pose2, worldPts):
    # Plot the base triangle
    TrianglePts = np.array([[0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [-1, 0, 1, 1],
                            [0, 0, 0, 1]])
    plt.plot(TrianglePts[:,0], TrianglePts[:,2], color='black')

    # Plot the four pose triangle
    color = ["red", "green", "blue", "brown"]
    for pose, c in zip(pose2, color):
        Pts = TrianglePts
        proj = np.concatenate([pose[1].T, -np.matmul(pose[1].T, pose[0][:, None])], axis=1).T
        Pts = np.matmul(Pts, proj)
        plt.plot(Pts[:,0], Pts[:,2], color=c)

    # Plot the four 3d points
    for x, c in zip(worldPts, color):
        plt.scatter(x[:,0], x[:,2], color=c, marker='.')

    plt.axis('equal')
    plt.show()



if __name__ == "__main__":
    """
    img1 = cv2.imread('1399381473388387.png', 0) # queryImage
    img2 = cv2.imread('1399381473450877.png', 0) # trainImage
    pt = GetSiftCorrespondence(img1, img2, 0.7)


    K = GetIntrinsic("data")
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    pts1 = np.float32([p[0] for p in pt])
    pts2 = np.float32([p[1] for p in pt])
    
    E, _ = cv2.findEssentialMat(pts1, pts2, focal=fx, pp=(cx,cy))
    poses = ExtractCameraPose(E)
    best_pose = DisambiguateCameraPose(poses, K, pt)
    #print(best_pose)

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx,cy))
    print("\n opencv")
    print(R)
    print(t)
    """


    filePath = "../Data/Imgs/"
    corDict = readMatches(filePath)
    curDict = corDict['12']

    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    savePath = "../Data/Imgs/GetInlierRANSAC/"

    for i in range(1, 2):
        for j in range(i+1, 3):
            if(len(corDict[str(i)+str(j)])) == 0:
                continue
            imgL = cv2.imread(filePath+str(i)+'.jpg')
            imgR = cv2.imread(filePath+str(j)+'.jpg')
            curDict = corDict[str(i)+str(j)]
            F, matchPts_inlier = GetInlierRANSAC(curDict, 50000, 0.0025)
            imgMatch = drawMatching(imgL, imgR, curDict, matchPts_inlier)

            cv2.imshow("Match"+str(i)+str(j), imgMatch)
            # cv2.imwrite(savePath + "Match"+str(i)+str(j) + '.jpg', imgMatch)

    cv2.waitKey()
    cv2.destroyAllWindows()
    # F, matchPts_inlier = GetInlierRANSAC(curDict, 50000, 0.0025)
    E = EssentialMatrixFromFundamentalMatrix(F, K)
    poses = ExtractCameraPose(E)
    best_pose = DisambiguateCameraPose(poses, matchPts_inlier, K)
