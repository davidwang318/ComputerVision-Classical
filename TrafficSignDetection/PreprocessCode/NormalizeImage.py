import cv2
import numpy as np
import os

def r_normalize(img):
    img = img.astype(np.float32)
    deno = np.sum(img, axis=2)
    deno += 10e-8
    mole = np.minimum(img[..., 2]-img[..., 0], img[..., 2]-img[..., 1])
    C = np.maximum(0, mole/deno)
    return (C*255).astype(np.uint8)


def b_normalize(img):
    img = img.astype(np.float32)
    deno = np.sum(img, axis=2)
    deno += 10e-8
    mole = img[..., 0] - img[..., 2]
    C = np.maximum(0, mole/deno)
    return (C*255).astype(np.uint8)


if __name__ == "__main__":

    for filename in sorted(os.listdir("frame_denoise")):
        img = cv2.imread("frame_denoise/" + filename)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_r = r_normalize(img)
        img_b = b_normalize(img)
        cv2.imshow("r", img_r)
        cv2.imshow("b", img_b)
        cv2.imwrite("frame_normalize_blue_0/" + filename, img_b)
        cv2.imwrite("frame_normalize_red_0/" + filename, img_r)
        
        cv2.waitKey(1)
    cv2.destroyAllWindows() 
    
