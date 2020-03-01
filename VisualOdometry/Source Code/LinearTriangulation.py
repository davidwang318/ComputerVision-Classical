import numpy as np
import cv2
from GetSiftCorrespondence import *
from GetIntrinsic import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *


def LinearTriangulation(pose1, pose2, K, corr):
    #proj1 = np.concatenate([np.identity(3), -pose1[0][:, None]], axis=1)
    #proj1 = K @ (pose1[1]) @ (proj1)
    #proj2 = np.concatenate([np.identity(3), -pose2[0][:, None]], axis=1)
    #proj2 = K @ (pose2[1]) @ (proj2)


    proj1 = np.matmul(K, (np.concatenate([pose1[1], pose1[0][:, None]], axis=1)))
    proj2 = np.matmul(K, (np.concatenate([pose2[1], pose2[0][:, None]], axis=1)))
  
    # Least square
    x1 = corr[:,0,0]
    y1 = corr[:,0,1]
    x2 = corr[:,1,0]
    y2 = corr[:,1,1]

    row1 = y1[:, None] * proj1[2] - proj1[1]
    row2 = proj1[0] - x1[:, None] * proj1[2]
    row3 = y2[:, None] * proj2[2] - proj2[1]
    row4 = proj2[0] - x2[:, None] * proj2[2]

    A = np.stack([row1, row2, row3, row4], axis=1)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    corr_3d = vh[:, -1]
    corr_3d /= corr_3d[..., -1][..., None]

    return corr_3d

if __name__ == "__main__":
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
    #print(poses[0][0].shape)

    pose_world = (np.zeros(3), np.identity(3))
    #print(np.zeros(3)[:, None].shape)
    #LinearTriangulation(pose_world, pose_world, pt)
    LinearTriangulation(pose_world, poses[3], K, pt)
