import cv2
import numpy as np
from ExtractCameraPose import *

# def PlotTrajectory(cor, pose, old_pose_R_cumulate):
#     C, R = pose[0], pose[1]
#     '''
#     Minus sign is for transferring relative motion of images (going backward) to
#     relative motion of camera (moving forward)
#     '''
#     cor_new = cor + old_pose_R_cumulate @ C
#
#     '''
#     By post-multiply the inverse of R = R(c' <- c):
#     R.T = R(c -< c')
#     we have
#     R(0 <- 1)R(1 <- 2) .... R(c <- c') = R(0 <- c')
#     which rotate the description of the translation vector C' back to
#     the description wrt the world coordinate
#     '''
#     old_pose_R_cumulate = old_pose_R_cumulate @ R
#
#     return cor_new, old_pose_R_cumulate

def PlotTrajectory(pose, H_cumulate):
    #print("ours, ", pose[0][2])
    C, R = pose[0], pose[1]
    C = C.reshape(-1,1)
    #H_rel = np.hstack((R.T, C))
    #if C[-1] < 0: C = -C
    #H_rel = np.hstack((R, -C))
    H_rel = np.matmul(R.T, (np.hstack((np.identity(3), -C))))
    H_rel[1, 0] = 0
    H_rel[1, 1] = 1
    H_rel[1, 2] = 0
    H_rel[1, 3] = 0
    H_rel = np.vstack((H_rel, np.array([0, 0, 0, 1])))
    H_cumulate = np.matmul(H_cumulate, H_rel)
    cor_new = H_cumulate[0][3], H_cumulate[2][3]
    
    return cor_new, H_cumulate


def PlotTrajectory_cv(pose, H_cumulate):
    #print("opencv, ", pose[0][2])
    C, R = pose[0], pose[1]
    C = C.reshape(-1,1)
    #H_rel = np.hstack((R, C))
    H_rel = np.matmul(R.T, (np.hstack((np.identity(3), -C))))
    H_rel = np.vstack((H_rel, np.array([0, 0, 0, 1])))
    H_cumulate = np.matmul(H_cumulate, H_rel)
    cor_new = H_cumulate[0][3], H_cumulate[2][3]

    return cor_new, H_cumulate

if __name__ == "__main__":

    img1 = cv2.imread('frame_gray/frame100.jpg', 0) # queryImage
    img2 = cv2.imread('frame_gray/frame101.jpg', 0) # trainImage
    pt = GetSiftCorrespondence(img1, img2, 0.7)
    F, _ = GetInlierRANSANC(pt, 20, 0.001)
    K = GetIntrinsic("Oxford_dataset/model")
    E = EssentialMatrixFromFundamentalMatrix(F, K)
    pose = ExtractCameraPose(E)
    cor_old = np.array([0, 0, 0])
    for i in range(len(pose)):
        cor_new = PlotTrajectory(cor_old, pose[i])
        print(i)
        print("cor: ", cor_new)

