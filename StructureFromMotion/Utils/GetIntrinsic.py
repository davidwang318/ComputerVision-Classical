import numpy as np


def GetIntrinsic(models_dir):
    intrinsics_path = models_dir + "/stereo_narrow_left.txt"
    intrinsics = np.loadtxt(intrinsics_path)
    # Intrinsics
    fx = intrinsics[0,0]
    fy = intrinsics[0,1]
    cx = intrinsics[0,2]
    cy = intrinsics[0,3]
    I_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) 

    return I_matrix
