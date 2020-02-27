#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Lih-Narn Wang (ytcdavid@terpmail.umd.edu)
Master of Engineering in Robotics
University of Maryland, College Park
"""


import sys
import numpy as np
import cv2
import argparse
from util import*

sys.dont_write_bytecode = True

def preProcess(img, fileName, outPath, nBest=200):
    """
    Input: RGB Image, Name of the image
    Output: RGB Image, Feature Points, Feature Descriptor
      Feature Points: pointNum*[x, y]
      Feature Descriptor: pointNum*featureNum
    """
    cv2.imshow('Original Image'+fileName, img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    """
    # Harris Corner Detection
    imgHarris = img.copy()
    imgGray32 = np.float32(imgGray)
    cornerHarris = cv2.cornerHarris(imgGray32,2,3,0.04)
    cornerHarris = cv2.dilate(cornerHarris,None)
    imgHarris[cornerHarris>0.01*cornerHarris.max()]=[0,0,255]
    cv2.imwrite(outPath+'CornerDetection/HarrisCorner_'+fileName, imgHarris)
    """

    # Corner Detection
    imgShi = img.copy()
    cornerShi = cv2.goodFeaturesToTrack(imgGray, 500, 0.01, 0)[:, 0 ,:] # n*1*2
    for x,y in cornerShi:
        cv2.circle(imgShi, (x,y), 3, (0,0,255), -1)
    cv2.imshow('Shi'+fileName, imgShi)

    # Perform ANMS
    imgShi2 = img.copy()
    shiBestPts = ANMS(cornerShi, nBest)
    for x,y in shiBestPts:
        cv2.circle(imgShi2, (x,y), 3, (0,0,255), -1)
    cv2.imshow('Shi_ANMS'+fileName, imgShi2)

    # Create the descriptor at the feature points
    imgShi3 = imgGray.copy()
    desShi = createDescriptor(imgShi3, shiBestPts, 40)

    return img, shiBestPts, desShi


def main():
    # Command line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InputPath', default='../Data/Train/')
    Parser.add_argument('--InputSet', default='Set1/')
    Parser.add_argument('--SavePath', default='Output/')
    Args = Parser.parse_args()

    # Set the input and output path
    inPath = Args.InputPath + Args.InputSet
    outPath = Args.SavePath + Args.InputSet
    inputImgs = sorted(os.listdir(inPath))
    print("Input image numbers: ", len(inputImgs))

    # Start the Autopano (1->2)
    for i in range(1, len(inputImgs)):
        if i == 1:
            fileName1 = inputImgs[0]
            img1 = cv2.imread(inPath+fileName1)
            # img1 = cv2.resize(img1, dsize=(300,300))
            Y, X, _ = img1.shape
            corOld = np.array([[[0, 0], [X-1, 0], [X-1, Y-1], [0, Y-1]]], dtype=np.float32)

        # Read the file
        fileName2 = inputImgs[i]
        img2 = cv2.imread(inPath+fileName2)
        # img2 = cv2.resize(img2, dsize=(300,300))

        # Extract the features and descriptors
        img1, pts1, des1 = preProcess(img1, fileName1, outPath, 200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img2, pts2, des2 = preProcess(img2, fileName2, outPath, 200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Match the features
        matchPts = featureMatching(pts1, des1, pts2, des2, 0.4)
        imgMatch1 = drawMatching(img1, img2, matchPts)
        showImg('Match1', imgMatch1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # RANSAC 
        H, matchPts = homoRANSAC(matchPts, 10, 50000)
        imgMatch2 = drawMatching(img1, img2, matchPts)
        showImg('Match2', imgMatch2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Blend the image
        imgBlend1, corOld = blendImg(img1, img2, corOld, H,matchPts, False)
        imgBlend, corOld = blendImg(img1, img2, corOld, H, matchPts, True)

        # Show the image
        showImg('Blend', imgBlend)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Update
        fileName1 = 'Merge'+str(i)+'.jpg'
        img1 = imgBlend
        # while(max(img1.shape) > 1200): img1 = cv2.resize(img1, dsize=(img1.shape[0]//2, img1.shape[1]//2))
        
if __name__ == '__main__':
    main()
 
