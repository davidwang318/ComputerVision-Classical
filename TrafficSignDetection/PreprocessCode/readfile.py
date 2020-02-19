import os
import cv2
from ReadCameraModel import *
from UndistortImage import *
from scipy import ndimage as ndi
from scipy import misc


index = 0
for filename in sorted(os.listdir("Oxford_dataset/stereo/centre")):
    img = cv2.imread("Oxford_dataset/stereo/centre/" + filename, -1)
    img_color = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel("Oxford_dataset/model")
    img_undistort = UndistortImage(img_color, LUT)
    img_gray = cv2.cvtColor(img_undistort, cv2.COLOR_BGR2GRAY)

    # Write Image
    cv2.imwrite("frame_gray/frame%d.jpg" %index, img_gray)
    index += 1

    # Show Image

    #cv2.imshow("bayer", img)
    #cv2.imshow("undistort", img_undistort)
    cv2.imshow("gray", img_gray)
    cv2.waitKey(1)

cv2.destroyAllWindows() 


