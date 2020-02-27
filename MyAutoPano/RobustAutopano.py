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
from Util.util import*

sys.dont_write_bytecode = True

def preProcess(img, fileName, outPath, n=1000, nBest=300):
    """
    Input: RGB Image, Name of the image
    Output: RGB Image, Feature Points, Feature Descriptor
      Feature Points: pointNum*[x, y]
      Feature Descriptor: pointNum*featureNum
    """
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Corner Detection
    imgShi = img.copy()
    cornerShi = cv2.goodFeaturesToTrack(imgGray, n, 0.01, 0)[:, 0 ,:] # n*1*2
    for x,y in cornerShi:
        cv2.circle(imgShi, (x,y), 3, (0,0,255), -1)

    # Perform ANMS
    imgShi2 = img.copy()
    shiBestPts = ANMS(cornerShi, nBest)
    for x,y in shiBestPts:
        cv2.circle(imgShi2, (x,y), 3, (0,0,255), -1)

    # Create the descriptor at the feature points
    imgShi3 = imgGray.copy()
    desShi = createDescriptor(imgShi3, shiBestPts, 40)

    return np.array([img, shiBestPts, desShi])


def sortImg(imgSet):
    # Initialize the relation table
    imgNum = len(imgSet)
    table = np.zeros((imgNum, imgNum))

    # Calculate the relation 
    for i in range(imgNum):
        pts1, des1 = imgSet[i][1:]
        for j in range(imgNum):
            print("caculate pair: ", (i, j))
            if (i==j): continue
            pts2, des2 = imgSet[j][1:]
            matchPts1 = featureMatching(pts1, des1, pts2, des2, 0.7)
            H, matchPts2 = homoRANSAC(matchPts1, 25, 1000)
            # Construct the realtion table
            table[i, j] = round(len(matchPts2)/float(len(matchPts1)), 3) 
    print("Relation table:\n", table)

    # Get the first image
    idxTable = np.argsort(-table, axis=1)
    idxRatio = np.zeros(imgNum)
    for i in range(imgNum):
        idxRatio[i] = table[i, idxTable[i,0]] / table[i, idxTable[i,1]]
    idxRank = np.argsort(-idxRatio)
    print("Index rank:\n", idxRank)

    # Back trck the orders
    imgOrder = -np.ones(imgNum)
    imgOrder[0] = idxRank[0]
    idx1 = idxRank[0]
    # imgOrder[0] = 0
    # idx1 = 0
    for i in range(1, imgNum):
        idx2 = 0
        while (imgOrder == idxTable[idx1,idx2]).any(): idx2+=1; 
        idx1 = imgOrder[i] = idxTable[idx1,idx2]
    print("Order of the image:\n", imgOrder)

    return imgOrder.astype(np.uint32)

def autoPano(imgSet, outPath):
    for i in range(len(imgSet)):
        if i == 0:
            fileName1 = str(i)+'.jpg'
            img1 = imgSet[i]
            # img1 = cv2.resize(img1, dsize=(300,600))
            Y, X, _ = img1.shape                                
            corOld = np.array([[[0, 0], [X-1, 0], [X-1, Y-1], [0, Y-1]]], dtype=np.float32)
            continue

        # Read the file
        fileName2 = str(i)+'.jpg'
        img2 = imgSet[i]
        # img2 = cv2.resize(img2, dsize=(300,600))
        # Extract the features and descriptors
        img1, pts1, des1 = preProcess(img1, fileName1, outPath, 2000, 1000)
        img2, pts2, des2 = preProcess(img2, fileName2, outPath, 2000, 1000)

        # Match the features
        matchPts = featureMatching(pts1, des1, pts2, des2, 0.7)
        imgMatch1 = drawMatching(img1, img2, matchPts)
        # cv2.imwrite(outPath+'Match1/'+str(i)+'.jpg', imgMatch1)

        # RANSAC 
        H, matchPts = homoRANSAC(matchPts, 10, 50000)
        imgMatch2 = drawMatching(img1, img2, matchPts)
        # cv2.imwrite(outPath+'Match2/'+str(i)+'.jpg', imgMatch2)

        # Blend the image
        #imgBlend1 = blendImg(img1, img2, H, matchPts, False)
        imgBlend, corOld = blendImg(img1, img2, corOld, H, matchPts, True)
        # cv2.imwrite(outPath+'Merge/'+str(i)+'.jpg', imgBlend)

        # Show the image
        showImg('Blend', imgBlend)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # Update
        fileName1 = 'Merge'+str(i)+'.jpg'
        img1 = imgBlend
        # while(max(img1.shape) > 4000): img1 = cv2.resize(img1, dsize=(img1.shape[0]//3*2, img1.shape[1]//3*2))

    return img1, corOld


def main():
    # Command line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--InputPath', default='../Data/Train/')
    Parser.add_argument('--InputSet', default='Set1/')
    Parser.add_argument('--SavePath', default='Output/')
    Args = Parser.parse_args()

    # Set the input and output path
    print("\n*** Set the input and output path")
    inPath = Args.InputPath + Args.InputSet
    outPath = Args.SavePath + Args.InputSet
    inputImgs = sorted(os.listdir(inPath))
    inputNum = len(inputImgs)
    print("Input image numbers: ", len(inputImgs))

    # Preprocess
    print("\n*** Pre process")
    imgSet = []
    imgInfoSet = []
    for i, inputImg in enumerate(inputImgs):
        imgProcess = cv2.imread(inPath+inputImg)
        imgSet.append(imgProcess)
        imgInfoSet.append(preProcess(imgProcess, inputImg, outPath+'preProcess/'))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    imgSet = np.array(imgSet)
    imgInfoSet = np.array(imgInfoSet)

    # Sort the input image
    print("\n*** Sort the images")
    imgOrder = sortImg(imgInfoSet)
    print(imgOrder)

    # Pair the input image
    mid = int(inputNum/2)
    if (inputNum%2): # odd
        imgSet1 = imgSet[imgOrder[:mid+1], ...]
        imgSet2 = np.flip(imgSet[imgOrder[mid:], ...], 0)
    else:
        imgSet1 = imgSet[imgOrder[:mid], ...]
        imgSet2 = np.flip(imgSet[imgOrder[mid:], ...], 0)

    img1, corOld = autoPano(imgSet1, outPath+'imgSet1/')
    img2, _ = autoPano(imgSet2, outPath+'imgSet2/')

    # Autopano for both sets:
    fileName1 = '1.jpg'
    fileName2 = '2.jpg'
    img1, pts1, des1 = preProcess(img1, fileName1, outPath+'final/', n=1500, nBest=500)
    img2, pts2, des2 = preProcess(img2, fileName2, outPath+'final/', n=1500, nBest=500)
    matchPts = featureMatching(pts1, des1, pts2, des2, 0.7)
    imgMatch1 = drawMatching(img1, img2, matchPts)
    showImg('Match1',imgMatch1)

    H, matchPts = homoRANSAC(matchPts, 10, 50000)
    imgMatch2 = drawMatching(img1, img2, matchPts)
    showImg('Match2',imgMatch2)

    imgMatch2 = drawMatching(img1, img2, matchPts)
    imgBlend, corOld = blendImg(img1, img2, corOld, H, matchPts, True)
    showImg('Blend', imgBlend)
    cv2.imwrite(outPath+'final.jpg', imgBlend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
 
