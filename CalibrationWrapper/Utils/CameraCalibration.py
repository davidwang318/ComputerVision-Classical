"""
    # CameraCalibration.py
    # Stereo Camera calibration
"""
import os
import math
import numpy as np
import cv2
import rigid

class CameraCalibration:
    def __init__(self):
        self.gridsize = (4,5)
        # self.gridsize = (9,6)
        self.M1 = self.M2 = self.d1 = self.d2 = None
        self.t = (0,0)
        self.angle = 0
        self.lmap1 = self.lmap2 = self.rmap1 = self.rmap2 = None

    def extract_points(self, base_dir):
        self.img_points_left = []
        self.img_points_right = []
        self.obj_points = []
        self.paths_left = []
        self.paths_right = []

        objp = np.zeros((self.gridsize[0]*self.gridsize[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.gridsize[0],0:self.gridsize[1]].T.reshape(-1,2)

        lfiles, rfiles = self.list_images(base_dir)

        for i,left_full_path in enumerate(lfiles):
            right_full_path = rfiles[i]

            left_name = os.path.basename(left_full_path)

            print("Processing " + left_name)
            lgray = cv2.imread(left_full_path, cv2.IMREAD_GRAYSCALE)
            rgray = cv2.imread(right_full_path, cv2.IMREAD_GRAYSCALE)

            lret, lcorners = cv2.findChessboardCorners(lgray, self.gridsize,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
            rret, rcorners = cv2.findChessboardCorners(rgray, self.gridsize,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            if not lret or not rret:
                print("Not enough points")
                continue

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            lcorners2 = cv2.cornerSubPix(lgray, lcorners, (11, 11), (-1, -1), criteria)
            rcorners2 = cv2.cornerSubPix(rgray, rcorners, (11, 11), (-1, -1), criteria)

            self.img_points_left.append(lcorners2)
            self.img_points_right.append(rcorners2)
            self.obj_points.append(objp)
            self.paths_left.append(left_full_path)
            self.paths_right.append(right_full_path)

        print("Total Valid Image Pairs : {}".format(len(self.obj_points)))
        self.imageSize = (lgray.shape[1], lgray.shape[0])

    def stereo_calibrate(self, numRadialDistortionCoefficients=2, estimateTangentialDistortion=False):
        print("Starting Stereo Calibration")

        term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

        flags = 0
        #flags |= cv2.CALIB_FIX_INTRINSIC
        #flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        #flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        #flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        #flags |= cv2.CALIB_FIX_ASPECT_RATIO
        if not estimateTangentialDistortion:
            flags |= cv2.CALIB_ZERO_TANGENT_DIST
        #flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        flags |= cv2.CALIB_RATIONAL_MODEL
        if numRadialDistortionCoefficients < 3:
            flags |= cv2.CALIB_FIX_K3
        if numRadialDistortionCoefficients < 4:
            flags |= cv2.CALIB_FIX_K4
        if numRadialDistortionCoefficients < 5:
            flags |= cv2.CALIB_FIX_K5
        if numRadialDistortionCoefficients < 6:
            flags |= cv2.CALIB_FIX_K6

        self.calibrationError, self.M1, self.d1, self.M2, self.d2, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            self.obj_points, self.img_points_left, self.img_points_right,
            None, None, None, None,
            imageSize=self.imageSize, criteria=term_crit, flags=flags)

        print('Mean reprojection error', self.calibrationError)
        print('Intrinsic_mtx_1', self.M1)
        print('dist_1', self.d1)
        print('Intrinsic_mtx_2', self.M2)
        print('dist_2', self.d2)
        print('R', self.R)
        print('T', self.T)
        print('E', self.E)
        print('F', self.F)

    def compute_undistortion_map(self):
        if self.M1 is None or self.M2 is None or self.d1 is None or self.d2 is None:
            print("Missing intrinsics parameters")
            return

        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(self.M1, self.d1, self.M2, self.d2, self.imageSize, self.R, self.T, alpha=1)
        print("R1")
        print(R1)
        print("R2")
        print(R2)
        print("P1")
        print(P1)
        print("P2")
        print(P2)
        print("Q")
        print(Q)
        print("roi_left")
        print(roi_left)
        print("roi_right")
        print(roi_right)

        self.lmap1, self.lmap2 = cv2.initUndistortRectifyMap(self.M1, self.d1, R1, P1,
                                                             self.imageSize, cv2.CV_16SC2)
        self.rmap1, self.rmap2 = cv2.initUndistortRectifyMap(self.M2, self.d2, R2, P2,
                                                             self.imageSize, cv2.CV_16SC2)

    def estimate_rigid(self, ref_frame = None):
        """Estimate rigid transform using one or all frames"""
        A = []
        B = []

        if ref_frame is None:
            num_images = len(self.img_points_left)
            for i in range(num_images):
                if i != 3:
                    continue
                lpoints_und = cv2.undistortPoints(self.img_points_left[i], self.M1, self.d1, P=self.M1)
                rpoints_und = cv2.undistortPoints(self.img_points_right[i], self.M2, self.d2, P=self.M2)
                A.append(lpoints_und)
                B.append(rpoints_und)
        else:
            i = int(ref_frame)
            lpoints_und = cv2.undistortPoints(self.img_points_left[i], self.M1, self.d1, P=self.M1)
            rpoints_und = cv2.undistortPoints(self.img_points_right[i], self.M2, self.d2, P=self.M2)
            A.append(lpoints_und)
            B.append(rpoints_und)

        A = np.array(A)
        numImages = A.shape[0]
        numPoints = A.shape[1]
        A = A.reshape((numImages*numPoints,2))
        A = np.matrix(A)
        A = A.T

        B = np.array(B)
        B = B.reshape((numImages*numPoints,2))
        B = np.matrix(B)
        B = B.T

        ret_R, self.t, self.angle = rigid.rigid_transform_2D(A,B)

    def save(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

        fs.writeComment("Stereo Camera Calibration")
        fs.write('M1', self.M1)
        fs.write('d1', self.d1)
        fs.write('M2', self.M2)
        fs.write('d2', self.d2)
        fs.write('R', self.R)
        fs.write('T', self.T)
        fs.write('E', self.E)
        fs.write('F', self.F)
        fs.write('calibrationError', self.calibrationError)
        fs.write('imageSize', self.imageSize)
        fs.write('gridsize', self.gridsize)

        fs.writeComment("Rigid Transform")
        fs.write('t', self.t)
        fs.write('angle', self.angle)
        fs.release()
        return True

    def load(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return False

        if fs.getNode("M1").empty() or fs.getNode("M2").empty():
            return False

        self.M1 = fs.getNode("M1").mat()
        self.d1 = fs.getNode("d1").mat()
        self.M2 = fs.getNode("M2").mat()
        self.d2 = fs.getNode("d2").mat()
        self.R = fs.getNode("R").mat()
        self.T = fs.getNode("T").mat()
        self.E = fs.getNode("E").mat()
        self.F = fs.getNode("F").mat()
        self.calibrationError = fs.getNode("calibrationError").real()
        imageSize = fs.getNode("imageSize").mat().flatten().tolist()
        self.imageSize = (int(imageSize[0]), int(imageSize[1]))
        print(self.imageSize)

        t = fs.getNode("t").mat().flatten().tolist()
        self.t = tuple(t)
        self.angle = fs.getNode("angle").real()
        gridsize = fs.getNode("gridsize").mat().flatten().tolist()
        self.gridsize = (int(gridsize[0]), int(gridsize[1]))
        fs.release()
        return True

    def list_images(self, base_dir):
        if os.path.isdir(os.path.join(base_dir, "left")):
            left_dir = os.path.join(base_dir, "left")
        if os.path.isdir(os.path.join(base_dir, "Left")):
            left_dir = os.path.join(base_dir, "Left")
        if os.path.isdir(os.path.join(base_dir, "right")):
            right_dir = os.path.join(base_dir, "right")
        if os.path.isdir(os.path.join(base_dir, "Right")):
            right_dir = os.path.join(base_dir, "Right")

        if right_dir is None or left_dir is None:
            return

        list_images = os.listdir(left_dir)
        list_images = [f for f in list_images if os.path.isfile(os.path.join(left_dir, f))]

        left_files = []
        right_files = []
        for left_name in list_images:
            # Find corresponding images in right_dir
            right_name = left_name
            hasPrefix = left_name[:4] == "left"
            if hasPrefix:
                right_name = "right" + left_name[4:]
            right_full_path = os.path.join(right_dir, right_name)
            if not os.path.isfile(right_full_path):
                continue

            left_full_path = os.path.join(left_dir, left_name)
            left_files.append(left_full_path)
            right_files.append(right_full_path)

        return left_files, right_files

    def stereo_image(self, left_path, right_path, compensateRigidTransform = True):
        """Compute anaglyph and SBS images"""

        if self.lmap1 is None or self.rmap1 is None:
            print("compute_undistortion_map")
            self.compute_undistortion_map()

        lima = cv2.imread(left_path)
        rima = cv2.imread(right_path)
        lima_und = cv2.remap(lima, self.lmap1, self.lmap2, cv2.INTER_LANCZOS4)
        rima_und = cv2.remap(rima, self.rmap1, self.rmap2, cv2.INTER_LANCZOS4)

        if compensateRigidTransform:
            c = math.cos(self.angle/2.)
            s = math.sin(self.angle/2.)
            ML = np.matrix([[c,-s,self.t[0]/2],[s,c,self.t[1]/2]], dtype=np.float64)
            lima_und = cv2.warpAffine(lima_und, ML, (lima_und.shape[1],lima_und.shape[0]))
            MR = np.matrix([[c,s,-self.t[0]/2],[-s,c,-self.t[1]/2]], dtype=np.float64)
            rima_und = cv2.warpAffine(rima_und, MR, (rima_und.shape[1],rima_und.shape[0]))

        if len(lima_und.shape) > 2 and lima_und.shape[2] == 3:
            lgray = cv2.cvtColor(lima_und, cv2.COLOR_BGR2GRAY)
        else:
            lgray = lima_und

        w = lima.shape[1]
        h = lima.shape[0]
        if len(rima_und.shape) > 2 and rima_und.shape[2] == 3:
            rgray = cv2.cvtColor(rima_und, cv2.COLOR_BGR2GRAY)
            sbs = np.zeros((h,w*2,lima.shape[2]), np.uint8)
        else:
            rgray = rima_und
            sbs = np.zeros((h,w*2), np.uint8)

        black = np.zeros(lgray.shape, np.uint8)
        anaglyph = cv2.merge((black,lgray,rgray))

        sbs[:,0:w] = lima_und
        sbs[:,w:w*2] = rima_und

        return anaglyph, sbs

def test_SBS(base_dir):
    cc = CameraCalibration()
    if not cc.load("matlab.yml"):
        print("Unable to open YAML file")
        return

    lfiles, rfiles = cc.list_images(base_dir)

    for i in range(len(lfiles)):
        left = lfiles[i]
        right = rfiles[i]
        basename = os.path.basename(lfiles[i])
        filename, extension = os.path.splitext(basename)
        print("Processing {}".format(basename))

        anaglyph, sbs = cc.stereo_image(left, right, False)
        cv2.imwrite(filename+"_anaglyph_test"+extension, anaglyph)
        cv2.imwrite(filename+"_SBS_test"+extension, sbs)

        cv2.imshow("Anaglyph", anaglyph)
        cv2.waitKey(0)

def test(base_dir):
    cc = CameraCalibration()
    cc.gridsize = (4,5)
    # cc.gridsize = (9,6)
    cc.extract_points(base_dir)
    cc.stereo_calibrate()
    cc.estimate_rigid()
    cc.compute_undistortion_map()
    cc.save("matlab.yml")

if __name__ == '__main__':
    # base_dir = "/Applications/MATLAB_R2019b.app/toolbox/vision/visiondata/calibration/stereo"
    base_dir = ""

    test(base_dir)
    test_SBS(base_dir)
