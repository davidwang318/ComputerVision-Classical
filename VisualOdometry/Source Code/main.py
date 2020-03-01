import cv2
import numpy as np
import matplotlib.pyplot as plt
from PlotTrajectory import *
from DisambiguateCameraPose import *


# Initialization
index = 50
img1 = cv2.imread("frame_gray/frame%d.jpg" %index)
index += 1
img2 = cv2.imread("frame_gray/frame%d.jpg" %index)
cor_1 = np.array([0, 0, 0])
cor_1_cv = np.array([0, 0, 0])
R_cumulate = np.identity(4)
R_cumulate_cv2 = np.identity(4)
old_best_pose = [[0, 0, 0], np.identity(3)]

# Intrinsic parameters
intrinsics = GetIntrinsic("Oxford_dataset/model/")
fx = intrinsics[0,0]
fy = intrinsics[1,1]
cx = intrinsics[0,2]
cy = intrinsics[1,2]

# Initialize GUI
plt.figure('trajectory')
plt.ion()
plt.grid()
plt.xlabel("axis x")
plt.ylabel("axis z")
plt.xlim((-100, 800))
plt.ylim((-300, 600))
plt.plot(0, 0, 'bo')

while True:

    matchPts = GetSiftCorrespondence(img1, img2, 10000, 0.6)
    
    """
    OpenCV implementation
    """

    E_cv2, mask1 = cv2.findEssentialMat(matchPts[:,0,:], matchPts[:,1,:], focal=fx, pp=(cx,cy), method=cv2.RANSAC, prob=0.999, threshold=0.5)
    _, R, t, mask2 = cv2.recoverPose(E_cv2, matchPts[:,0,:], matchPts[:,1,:], focal=fx, pp=(cx,cy))
    if np.linalg.det(R) < 0: R = -R
    pose_cv2 = [t, R]   
    cor_2_cv, R_cumulate_cv2 = PlotTrajectory_cv(pose_cv2, R_cumulate_cv2)

    """
    My implementation
    """

    # Estimate the Fundamental Matrix
    F, inPts = GetInlierRANSAC(matchPts, 20000, 0.005) # _, iteration, threshold
    E = EssentialMatrixFromFundamentalMatrix(F)
    poses = ExtractCameraPose(E)

    # Error rejection
    if len(inPts) > 16:
        best_pose, _, _ = DisambiguateCameraPose(poses, inPts)
    else: 
        print("Poor Fundamental Matrix")
        best_pose = old_best_pose
    
    # Plot Trajectory
    cor_2, R_cumulate = PlotTrajectory(best_pose, R_cumulate)
    plt.plot([cor_1[0], cor_2[0]], [cor_1[1], cor_2[1]], "g", linewidth=2)
    plt.plot([cor_1_cv[0], cor_2_cv[0]], [cor_1_cv[1], cor_2_cv[1]], "r", linewidth=2)
    plt.show()
    plt.pause(0.01)

    # Show current frame
    cv2.imshow("img1", img1)
    if index > 3868:
        cv2.waitKey(0)
    else:
        cv2.waitKey(1) 

    # Update
    old_best_pose = best_pose
    cor_1 = cor_2
    cor_1_cv = cor_2_cv
    index += 2
    img1 = img2
    img2 = cv2.imread("frame_gray/frame%d.jpg" %index, 0)
    # img2 = cv2.equalizeHist(img2)

