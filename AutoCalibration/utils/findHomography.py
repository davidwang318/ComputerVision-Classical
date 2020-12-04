import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import sys


def createModelPoints(gridSize=(9,6), scale=20):
    x, y = np.meshgrid(range(gridSize[0]), range(gridSize[1]))
    x = x.reshape(54, 1)
    y = y.reshape(54, 1)
    mdPts = np.concatenate((x,y), axis=1)*scale
    mdPts = np.asarray(mdPts)
    return mdPts

def extractImagePoints(filePaths, gridSize=(9,6), show=False):
    imgPts = []
    for filePath in filePaths:
        print(">>>> extractSensorPoints: " + filePath)
        img = cv2.imread(filePath)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(imgGray, gridSize)

        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(imgGray, corners, (11,11), (-1,-1), criteria)
        imgPts.append(corners[:, 0, :])
        
        if show:
            cv2.drawChessboardCorners(img, gridSize, corners, ret)
            cv2.imshow("extractPoints", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return ret, np.array(imgPts)

def normalizePoints(pts, mode=1):
    xyMean = np.mean(pts, axis=0)
    n = pts.shape[0]

    if mode == 1:
        sx = sy = (2**0.5) * n / np.sum(np.linalg.norm(pts-xyMean, axis=1))
    elif mode == 2:
        lamda_x = (1.0/n) * np.sum(np.power(pts[:, 0] - xyMean[0], 2))
        lamda_y = (1.0/n) * np.sum(np.power(pts[:, 1] - xyMean[1], 2))
        sx = (2.0 / lamda_x) ** 2
        sy = (2.0 / lamda_y) ** 2
    else:
        print(">>>> Warning! Choose the correct mode in normalizePoints function")
        sys.exit()

    N = np.array([[sx, 0, -sx*xyMean[0]],
                  [0, sy, -sy*xyMean[1]],
                  [0,  0, 1]])
    N_inv = np.array([[1/sx, 0, xyMean[0]],
                      [0, 1/sy, xyMean[1]],
                      [0, 0, 1]])

    homoAxis = np.ones(n).reshape((n,1))
    homoPts = np.concatenate((pts, homoAxis), axis=1)
    normPts = np.dot(homoPts, N.T)[:, :2]

    return normPts, N, N_inv

def findHomography(mdPts, imgPts, normalize=True, findMode=1):
    
    if normalize:
        mdPts_norm, mdN, _ = normalizePoints(mdPts, findMode)
        imgPts_norm, imgN_inv = [], []
        for i, imgPt in enumerate(imgPts):
            imgPts_tmp, _, imgN_inv_tmp = normalizePoints(imgPt, findMode)
            imgN_inv.append(imgN_inv_tmp)
            imgPts_norm.append(imgPts_tmp)
        imgN_inv = np.array(imgN_inv)
        imgPts_norm = np.array(imgPts_norm)
    else:
        mdPts_norm = np.copy(mdPts)
        imgPts_norm = np.copy(imgPts)

    H_all = []
    homoAxis = np.ones(54).reshape((54,1))
    for i, imgPt_norm in enumerate(imgPts_norm):
        H, _ = cv2.findHomography(mdPts_norm, imgPt_norm)
        tmp = np.concatenate((mdPts_norm, homoAxis), axis=1)
        tmp = np.dot(tmp, H.T)[:, :2]
        if normalize:
            H = np.matmul(np.matmul(imgN_inv[i], H), mdN)
        H_all.append(H)

    return H_all




if __name__ == '__main__':
    files = sorted(glob.glob('Calibration_Imgs/*.jpg'))
    gridSize = (9, 6)

    # """
    # Testing createModelPoints()
    mdPts = createModelPoints()

    # print(mdPts.shape)
    # """

    # """
    # Testing extractImagePoints()
    _, imgPts = extractImagePoints(files)
    # print(imgPts.shape)

    # """

    # Testing normalizePoints()
    # _, _, _ = normalizePoints(mdPts)

    # Testing findHomography()
    H = findHomography(mdPts, imgPts)
    homoAxis = np.ones(54).reshape((54,1))
    mdPts = np.concatenate((mdPts, homoAxis), axis=1)
    err = 0
    for i, h in enumerate(H):
        print("    %d" % i)
        tmp = np.dot(mdPts, h.T)
        tmp /= tmp[:,2][...,None]
        tmp = tmp[:, :2]
        err += np.mean(np.linalg.norm(tmp-imgPts[i], axis=1))
    err /= 13.0
    print(err)


