import cv2
import numpy as np
import os


if __name__ == '__main__':

    index = 0
    for filename in sorted(os.listdir("input/")):
        img = cv2.imread("input/" + filename)
        img_denoise = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
        # Write Image
        cv2.imwrite("frame_denoise/frame%d.jpg" %index, img)
        index += 1

        # Show Image
        cv2.imshow("frame", img)
        cv2.imshow("denoise", img_denoise)
        cv2.waitKey(1)

    cv2.destroyAllWindows() 
