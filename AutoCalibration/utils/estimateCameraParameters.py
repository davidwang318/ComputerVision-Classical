import numpy as np
from utils.findHomography import *
from utils.utils import *

def v(H, p, q):
    return np.array([[H[0,p] * H[0,q],
                      H[0,p]*H[1,q] + H[1,p]*H[0,q],
                      H[1,p] * H[1,q],
                      H[2,p]*H[0,q] + H[0,p]*H[2,q],
                      H[2,p]*H[1,q] + H[1,p]*H[2,q],
                      H[2,p] * H[2,q]]])

def homographyRow(H):
    row1 = v(H, 0, 1)
    row2 = v(H, 0, 0) - v(H, 1, 1)
    return np.concatenate((row1, row2), axis=0)

def intrinsicHealper(B):
    w = B[0]*B[2]*B[5] - B[1]**2*B[5] - B[0]*B[4]**2 + 2*B[1]*B[3]*B[4] - B[2]*B[3]**2
    d = B[0]*B[2] - B[1]**2
    alpha = ( w / (d*B[0]) )**0.5
    beta = ( w / d**2 * B[0])**0.5
    gamma = B[1] * ( w / (d**2*B[0]) )**0.5
    u0 = (B[1]*B[4] - B[2]*B[3]) / d
    v0 = (B[1]*B[3] - B[0]*B[4]) / d

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])

    return K

def extrinsicHealper(K, H, trueR):
    K_inv = np.linalg.inv(K)
    P = []
    for h in H:
        lambda_ = 1 / np.linalg.norm((np.dot(K_inv, h[:, 0])))
        r0 = lambda_ * np.dot(K_inv, h[:, 0])
        r1 = lambda_ * np.dot(K_inv, h[:, 1])
        t = lambda_ * np.dot(K_inv, h[:, 2])[:, None]
        
        r2 = np.cross(r0, r1)
        Q = np.stack((r0, r1, r2), axis=1)

        if trueR:
        	u, s, vh = np.linalg.svd(Q)
        	R = np.matmul(u, vh)
        else:
        	R = Q
        P_tmp = np.concatenate((R, t), axis=1)
        P.append(P_tmp)

    return np.array(P)


def estimateCameraParameter(H, trueR=True):
    A = []
    for h in H:
        row_tmp = homographyRow(h)
        A = np.concatenate((A, row_tmp), axis=0) if len(A) > 0 else row_tmp
    u, s, vh = np.linalg.svd(A)
    B = vh[np.argmin(s)]

    K = intrinsicHealper(B)

    P = extrinsicHealper(K, H, trueR)

    return K, P	


if __name__ == "__main__":

    files = sorted(glob.glob('Calibration_Imgs/*.jpg'))
    gridSize = (9, 6)

    mdPts = createModelPoints()

    _, imgPts = extractImagePoints(files)

    H = findHomography(mdPts, imgPts, normalize=True, findMode=2)

    K, P = estimateCameraParameters(H, False)
    
    print(reprojectError(mdPts, imgPts, K, P))
