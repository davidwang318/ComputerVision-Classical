import numpy as np
import cv2
from GetSiftCorrespondence import *
from GetInlierRANSANC import * 
from GetIntrinsic import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *

def DisambiguateCameraPose(poses, corr):
    K = GetIntrinsic("data")
    pose_world = (np.zeros(3), np.identity(3))
    best_pose = pose_world
    n = 0
    
    for pose in poses:
        X = LinearTriangulation(pose_world, pose, K, corr)[..., :-1]
        Rz = (pose[1])[-1]

        cheirality = np.matmul(X, (Rz)) + pose[0][2] > 0
        cheirality &= X[:, 2] > 0

        z = X[cheirality, :]
        if z.shape[0] > n:
            n = z.shape[0]
            best_pose = pose
            best_point = z
            best_inlier = corr[cheirality, :, :]

    return best_pose, best_point, best_inlier

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
    best_pose = DisambiguateCameraPose(poses, K, pt)
    #print(best_pose)

    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=fx, pp=(cx,cy))
    print("\n opencv")
    print(R)
    print(t)
