import numpy as np
import cv2
from GetInlierRANSANC import*

def GetSiftCorrespondence(img1, img2, numPts, ratio):
    """
    Input: two images and the ration to decide a good match.
    Ourput: n*2*2 = num_points * (img1, img2) * (x, y)
    """
    print("*** SIFT process:")

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create(numPts)

    #ints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good = []
    good_pt = []
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)
            good_pt.append([kp1[m.queryIdx].pt, kp2[m.trainIdx].pt]) 
    good_pt = np.array(good_pt)

    print('Number of feature matches: ', good_pt.shape[0])
    print("==========")

    """
    # Use RANSAC to get the best model and inliers
    F, matchPts = GetInlierRANSAC(good_pt, 5000, 0.005)

    # Show image
    draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=(255,0,0),
                       matchesMask=matchesMask, flags=0) 
    imgMatch1 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    cv2.imshow('Matches', imgMatch1)
    imgMatch2 = drawMatching(img1, img2, matchPts)
    cv2.imshow('Matches2', imgMatch2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return good_pt

if __name__ == "__main__":
    img1 = cv2.imread('frame_gray/frame100.jpg', 0) # queryImage
    img2 = cv2.imread('frame_gray/frame101.jpg', 0) # trainImage
    pt = sift(img1, img2, 0.4)

