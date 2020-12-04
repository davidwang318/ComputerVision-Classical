from utils import *
from scipy import optimize


def errorFunction(para, mdPts, imgPts):
    K = np.array([[para[0], para[1], para[3]],
                  [0,       para[2], para[4]],
                  [0,     0,     1]])
    k1, k2 = para[5:7]
    P = []
    for i in range(imgPts.shape[0]):
        start = 7 + 6*i
        mid = start + 3
        end = start + 6
        R, _ = cv2.Rodrigues(para[start:mid][:, None])
        T = para[mid:end].reshape((3, 1))
        P_ = np.concatenate((R, T), axis=1)
        P.append(P_)
        start = end
    P = np.array(P)
    err = reprojectError(mdPts, imgPts, K, P, mode='ssd', distortion=[k1, k2])
    # err = reprojectError(mdPts, imgPts, K, P, distortion=[k1, k2])
    # print("error in lsq: %f" %err)
    return err


def lsqOptimization(mdPts, imgPts, K, P):
    parameters = np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], 0, 0])
    for p in P:
        r_rod, _ = cv2.Rodrigues(p[:,:3])
        t = p[:,3]
        parameters = np.concatenate((parameters, r_rod.ravel()))
        parameters = np.concatenate((parameters, t))
    
    res_lsq = optimize.minimize(errorFunction, parameters, args=(mdPts, imgPts))
    
    para = res_lsq.x
    K_new = np.array([[para[0], para[1], para[3]],
                  [0,       para[2], para[4]],
                  [0,     0,     1]])
    distortion = para[5:7]
    P_new = []
    for i in range(imgPts.shape[0]):
        start = 7 + 6*i
        mid = start + 3
        end = start + 6
        R, _ = cv2.Rodrigues(para[start:mid][:, None])
        T = para[mid:end].reshape((3, 1))
        P_ = np.concatenate((R, T), axis=1)
        P_new.append(P_)
        start = end
    P_new = np.array(P_new)

    return K_new, P_new, distortion


if __name__ == "__main__":

    files = sorted(glob.glob('Calibration_Imgs/*.jpg'))
    gridSize = (9, 6)

    mdPts = createModelPoints()

    _, imgPts = extractImagePoints(files)

    H = findHomography(mdPts, imgPts)

    K, P = estimateCameraParameters(H)
    err = reprojectError(mdPts, imgPts, K, P)
    print('original error: %f' %err)

    new_K, new_P, distortion = lsqOptimization(mdPts, imgPts, K, P)

    err = reprojectError(mdPts, imgPts, new_K, new_P, distortion=distortion)

    print('Final error: %f' %err)



