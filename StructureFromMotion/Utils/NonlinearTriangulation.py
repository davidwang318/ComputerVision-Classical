import numpy as np
import cv2
import matplotlib.pyplot as plt
from GetInlierRANSANC import * 
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from DisambiguateCameraPose import *
from scipy import optimize

def ReprojectionError(worldPts, imgPts, proj):
    # Restore worldPts
    worldPts = worldPts.reshape((-1, 3))
    # Extract the projection matrix (have K inside)
    proj1 = proj[0]
    proj2 = proj[1]
    # Construct the homogeneous 3d pts
    homoAxis = np.ones((len(worldPts), 1))
    worldPts_homo = np.concatenate((worldPts, homoAxis), axis=1)
    # Project the 3d pts to image coordinates
    projPts1 = np.matmul(worldPts_homo, proj1.T)
    projPts2 = np.matmul(worldPts_homo, proj2.T)
    # Normalize the image pts
    projPts1 /= projPts1[..., -1][..., None]
    projPts2 /= projPts2[..., -1][..., None]
    # Remove the homogeneous axis
    projPts1 = projPts1[..., :2]
    projPts2 = projPts2[..., :2]
    # Calculate the reprojection error
    diff1 = projPts1 - imgPts[:, 0, :2]
    diff2 = projPts2 - imgPts[:, 1, :2]
    # reprojError = np.sum(np.linalg.norm(diff1, axis=1)**2 + np.linalg.norm(diff2, axis=1)**2)
    reprojError = np.sum(np.linalg.norm(diff1, axis=1) + np.linalg.norm(diff2, axis=1))

    return reprojError

def NonlinearTriangulation(worldPts, imgPts, best_pose, K):
    print("*** NonlinearTriangulation process:")
    # Construct the projection matrix of two perspective
    pose_world = (np.zeros(3), np.identity(3))
    proj1 = np.matmul(K, (np.concatenate([pose_world[1], pose_world[0][:, None]], axis=1)))
    proj2 = np.matmul(K, (np.concatenate([best_pose[1], best_pose[0][:, None]], axis=1)))

    # Flatten the world points
    worldPts = worldPts.flatten()

    # Nonlinear Triangulation
    print("Original Mean ReprojectionError: ", ReprojectionError(worldPts, imgPts, [proj1, proj2]) / (2*len(worldPts)))
    print("Optimizing......")
    nonlinearRes = optimize.minimize(ReprojectionError, worldPts, args=(imgPts, [proj1, proj2]), options={'maxiter': 1000, 'disp': True})
    print("Optimized Mean ReprojectionError: ", ReprojectionError(nonlinearRes.x, imgPts, [proj1, proj2]) / (2*len(worldPts)))

    # Get the refined 3D points    
    worldPts_refined = nonlinearRes.x.reshape((-1,3))

    return worldPts_refined

def PlotComparison(best_pose, best_point, best_point_refined):
    # Plot the base triangle
    TrianglePts = np.array([[0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [-1, 0, 1, 1],
                            [0, 0, 0, 1]])
    plt.plot(TrianglePts[:,0], TrianglePts[:,2], color='black')

    # Plot the pose triangle
    Pts = TrianglePts
    proj = np.concatenate([best_pose[1].T, -np.matmul(best_pose[1].T, best_pose[0][:, None])], axis=1).T
    Pts = np.matmul(Pts, proj)
    plt.plot(Pts[:,0], Pts[:,2], color='red')

    # Plot the original/refined 3d points
    plt.scatter(best_point[:,0], best_point[:,2], color="red", marker='.')
    plt.scatter(best_point_refined[:,0], best_point_refined[:,2], color="green", marker='.')

    plt.axis('equal')
    plt.show()

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

    best_point_refined = NonlinearTriangulation(best_point, best_inlier, best_pose, K)
    PlotComparison(best_pose, best_point, best_point_refined)

