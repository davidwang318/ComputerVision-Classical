import cv2
import numpy as np
from numpy import linalg as la
from sift import *

def normalizeFeature(pts):
    nums_ = len(pts)
    mean_ = np.mean(pts, axis=0)
    scale_ = np.sum(la.norm(pts-mean_, axis=1) / 2**0.5) / nums_
    scale_ = 1/ scale_
    T = np.array([[scale_, 0, -mean_[0]*scale_], 
                     [0, scale_, -mean_[1]*scale_],
                     [0, 0, 1]])
    newPts = np.matmul(T, pts.T).T
    return newPts, T


def EstimateFundamentalMatrix(x1, x2):
    """
    Input: more than 8 piars of matching points, [[x1, y1, 1], ...], n*3
    Output: Fundamental matrix, 3*3
    """
    p_num = len(x1)
    
    # Center and Normalize coordinates:
    X1, T1 = normalizeFeature(x1)
    X2, T2 = normalizeFeature(x2)

    # Using SVD to solve AX=0
    A = np.zeros((p_num, 9))
    for i in range(p_num):
        A[i] = [X1[i][0]*X2[i][0], X1[i][0]*X2[i][1], X1[i][0],
                X1[i][1]*X2[i][0], X1[i][1]*X2[i][1], X1[i][1],
                X2[i][0], X2[i][1], 1]
    u, s, vh = np.linalg.svd(A)
    parameter_f = vh[-1]
    F_normal = parameter_f.reshape((3,3)).T

    # Force rank(F) = 2
    u_f, s_f, vh_f = np.linalg.svd(F_normal, full_matrices=True)
    s_f[-1] = 0
    F_normal_2 = np.matmul(u_f, (s_f[..., None] * vh_f))
    F = np.matmul(np.matmul(T2.T, F_normal_2), T1)
    F /= np.linalg.norm(F)
    F /= F[2][2]

    return F

def ransac(pt, M):

    # Homogenious coordinate
    ones = np.ones((len(pt), 2, 1))
    pt_homo = np.concatenate((np.array(pt), ones), axis=2)

    # Randomly pick 8 points
    indeces = np.random.choice(len(pt), 15)
    xs1 = np.array([pt_homo[idx, 0, :] for idx in indeces])
    xs2 = np.array([pt_homo[idx, 1, :] for idx in indeces])

    # Estimate fundamental matrix from 8 points
    F = np.zeros((3, 3))
    #F = EstimateFundamentalMatrix(xs1, xs2)

    
    score = np.sum(np.matmul(pt_homo[:, 1, :], F)*pt_homo[:, 0, :], axis=1)
    
    return xs1, xs2

if __name__ == "__main__":
    img1 = cv2.imread('frame_gray/frame100.jpg', 0) # queryImage
    img2 = cv2.imread('frame_gray/frame101.jpg', 0) # trainImage
    pt = sift(img1, img2, 0.7)
    xs1, xs2 = ransac(pt, 10)
    print(xs1, xs2)
    F1 = EstimateFundamentalMatrix(xs1, xs2)
    F2,_ = cv2.findFundamentalMat(xs1,xs2,cv2.FM_8POINT)
    print("My Result:")
    print(F1)
    print("Opencv 8 point Result:")
    print(F2)
    # for i in range(3):
    #     print(np.linalg.matrix_rank(F_cv[i*3:i+3]))
    # score = np.sum(np.matmul(xs2, F)*xs1, axis=1)

