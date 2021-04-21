"""
Lih-Narn Wang (ytcdavid@terpmail.umd.edu)
M.Eng in Robtics,
University of Maryland, College Park
------------------------------------
A wrapper function to easily do the calibration and 
save the parameters into a .json file.
"""

from Utils.CameraCalibration import *
from Utils.rigid import *
from Utils.healper import *
import argparse

def main(base_dir, file_name):

    CalibrationObj = CameraCalibration()
    CalibrationObj.gridsize = (4,5)
    CalibrationObj.extract_points(base_dir)
    CalibrationObj.stereo_calibrate()
    CalibrationObj.estimate_rigid()
    CalibrationObj.compute_undistortion_map()
    CalibrationObj.save("calibrationResult.yml")

    l1, l2 = load_yml("calibrationResult.yml")
    saveInJSON(file_name, l1, l2)


if __name__ == '__main__':
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default="Calibration_Imgs/", help="BasePath of the calibration image folders")
    Parser.add_argument('--SaveFilePath', default='Output/', help="Path to save the .json file")
    Parser.add_argument('--SaveFileName', default="camera_parameters_binning.json", help="The file name of the parameters, should be ***.json")

    Args = Parser.parse_args()
    baseDir = Args.BasePath
    fileName = Args.SaveFilePath + Args.SaveFileName 

    main(baseDir, fileName)
