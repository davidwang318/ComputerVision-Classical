import cv2
import numpy as np


def imadjust(img, tol=1, out=[0, 255]):
    img_contrast = np.zeros(img.shape).astype(np.uint8)

    for index in range(img.shape[2]):
        img_tmp = img[..., index]
        tol = max(0, min(100, tol))
        if (tol > 0):
            # Histogram
            hist = cv2.calcHist([img_tmp],[0],None,[256],[0,256])
            # Cumulative histogram
            cum = np.zeros(hist.shape)
            for i in range(1, len(hist)):
                cum[i] = cum[i-1] + hist[i]
            # Compute bounds
            total = img_tmp.shape[0]*img_tmp.shape[1]
            low_bound = int(total * tol / 100)
            upp_bound = int(total * (100-tol) / 100)
            i,_ = np.where(cum >= low_bound)
            j,_ = np.where(cum >= upp_bound)
            bound_ind = np.array([i[0], j[0]])
            # print(bound_ind)
        # Stretching
        scale = (out[1] - out[0])/(bound_ind[1]-bound_ind[0])
        r, c = 0, 0
        for r in range(img_tmp.shape[0]):
            for c in range(img_tmp.shape[1]):
                vs = max(img_tmp[r][c] - bound_ind[0], 0)
                vd = min(int(vs * scale + 0.5) + out[0], out[1])
                if vd > 255: vd = 255
                if vd < 0: vd = 0
                img_contrast[r][c][index] = vd

    return img_contrast


if __name__ == '__main__':
    tmp = 0
    index = tmp
    img = cv2.imread("frame_denoise/frame%d.jpg" %index)
    while index <= 2860:
        # Write Image
        cv2.imwrite("frame_denoise/frame%d.jpg" %index, img_denoise)
        dst = imadjust(img, 10)

        # Show Image
        cv2.imshow("contrast", dst)
        cv2.imshow("frame", img)
        cv2.waitKey(1)

        # Save Image
        cv2.imwrite("frame_contrast_10/frame%d.jpg" %index, dst)

        # Read Image
        index += 1
        img = cv2.imread("frame_denoise/frame%d.jpg" %index)

    cv2.destroyAllWindows() 
