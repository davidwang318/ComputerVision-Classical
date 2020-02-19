import cv2
import numpy as np
import os

index = 2600
while True:
    # Input image
    #img_b = cv2.imread("frame_normalize_blue_0/frame%d.jpg" %index, -1)
    img_r = cv2.imread("frame_normalize_red_0/frame%d.jpg" %index, -1)
    img_rgb = cv2.imread("Loose_Red_result/%d.jpg" %index)
    img_tem = cv2.imread("frame_denoise/frame%d.jpg" %index)
    images = [img_b]
    
    mask = np.zeros(img_b.shape).astype(np.uint8)

    for img in images:
        # Region proposals based on MSER feature
        mser = cv2.MSER_create(_min_area=300, _max_area=10000, _max_variation=0.5, _min_diversity=0.2, _delta=2)
        regions = np.array(mser.detectRegions(img)[0])
        hulls = [cv2.convexHull(p) for p in regions]

        for p in hulls:
            epsilon = 0.008 * cv2.arcLength(p, True)
            approx = cv2.approxPolyDP(p, epsilon, True)
            x, y, w, h = cv2.boundingRect(p)
            if 0.6 <= w/h <= 1.4 and 200 < w*h < 12000 and y < 700:
                cv2.rectangle(mask, (x, y), (x+w,y+h), 1, -1)
    img_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    cv2.imwrite("blue_hsv_mser/frame%d.jpg" %index, img_rgb)
    cv2.imshow('img', img_rgb)
    cv2.imshow('template', img_tem)
    index += 1
    cv2.waitKey(1)
cv2.destroyAllWindows()
