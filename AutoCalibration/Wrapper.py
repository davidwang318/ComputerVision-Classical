from utils.findHomography import *
from utils.estimateCameraParameters import *
from utils.nonlinearOptimization import *
from utils.utils import *
import argparse
import glob


def wrapper(imgPath, savePath, findNormal, findMode, trueR):
    files = sorted(glob.glob(imgPath))

    mdPts = createModelPoints(scale=21.5)

    _, imgPts = extractImagePoints(files)

    H = findHomography(mdPts, imgPts, findNormal, findMode)

    K, P = estimateCameraParameter(H, trueR)

    err = reprojectError(mdPts, imgPts, K, P)
    print('original error: %f' %err)

    new_K, new_P, distortion = lsqOptimization(mdPts, imgPts, K, P)

    err, pts_final = reprojectError(mdPts, imgPts, new_K, new_P, distortion=distortion, returnPts=True)

    print('Final error: %f' %err)
    print('--------------------')
    print('Estimated Intrinsic Camera Parameters: ')
    print(new_K)
    print('Estimated Distortion Parameters: [k1, k2]')
    print(distortion)

    undistortImage(files, savePath, new_K, distortion, pts_final)


if __name__ == '__main__':
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--ImagePath', default='Calibration_Imgs/', help='Base path of images, Default:Calibration_Imgs/')
    # Parser.add_argument('--SavePath', default='Output/undistort_Imgs/', help='Base path to store the undistort images')
    # Parser.add_argument('--Normalize', type=bool, default=False, help='Normalize the points to find homography')
    # Parser.add_argument('--NormalizeMode', type=int, default=1, help='Normalize Mode. Only activate when Normalize option is True')
    # Parser.add_argument('--ConvertR', type=bool, default=True, help='Convert the estimate Rotation into a valid Rotation')

    # Args = Parser.parse_args()
    # imgPath = Args.ImagePath + '*.jpg'
    # savePath = Args.SavePath
    # Normalize = Args.Normalize
    # NormalizeMode = Args.NormalizeMode
    # TrueR = Args.ConvertR
    
    # wrapper(imgPath, savePath, Normalize, NormalizeMode, TrueR)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp *= 21.5

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('Calibration_Imgs/*.jpg')

    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print(mtx)
    print(dist)
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error/len(objpoints))