from utils.findHomography import *
from utils.estimateCameraParameters import *
from utils.nonlinearOptimization import *
from utils.utils import *
import argparse


def wrapper(imgPath, savePath, findNormal, findMode, trueR):
    files = sorted(glob.glob(imgPath))

    mdPts = createModelPoints(scale=21.5)

    _, imgPts = extractImagePoints(files)

    H = findHomography(mdPts, imgPts, findNormal, findMode)

    K, P = estimateCameraParameters(H, trueR)

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
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ImagePath', default='Calibration_Imgs/', help='Base path of images, Default:Calibration_Imgs/')
    Parser.add_argument('--SavePath', default='Output/undistort_Imgs/', help='Base path to store the undistort images')
    Parser.add_argument('--Normalize', type=bool, default=False, help='Normalize the points to find homography')
    Parser.add_argument('--NormalizeMode', type=int, default=1, help='Normalize Mode. Only activate when Normalize option is True')
    Parser.add_argument('--ConvertR', type=bool, default=True, help='Convert the estimate Rotation into a valid Rotation')

    Args = Parser.parse_args()
    imgPath = Args.ImagePath + '*.jpg'
    savePath = Args.SavePath
    Normalize = Args.Normalize
    NormalizeMode = Args.NormalizeMode
    TrueR = Args.ConvertR
    
    wrapper(imgPath, savePath, Normalize, NormalizeMode, TrueR)