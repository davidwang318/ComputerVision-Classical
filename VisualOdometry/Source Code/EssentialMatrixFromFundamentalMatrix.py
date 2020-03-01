import numpy as np
import cv2
from GetSiftCorrespondence import *
from GetInlierRANSANC import * 
from GetIntrinsic import *


def EssentialMatrixFromFundamentalMatrix(F):
    print("*** Essential matrix process:")
    K = GetIntrinsic("data")
    E = np.matmul(np.matmul(K.T, F), K)
    u, s, vh = np.linalg.svd(E)
    newValues = np.ones(s.shape)
    newValues[-1] = 0
    reconstE = np.matmul(u, (newValues[..., None]*vh))
    print(reconstE)
    print("==========")
    return reconstE

if __name__ == "__main__":
    img1 = cv2.imread('frame_gray/frame100.jpg') # queryImage
    img2 = cv2.imread('frame_gray/frame101.jpg') # trainImage
    pt = GetSiftCorrespondence(img1, img2, 0.7)
    F, _ = GetInlierRANSAC(pt, 5000, 0.001)
    E = EssentialMatrixFromFundamentalMatrix(F)

    print(E)
