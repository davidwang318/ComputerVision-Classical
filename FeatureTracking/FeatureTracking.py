import cv2
import numpy as np
import argparse
from InverseLK import*


# Parse Command Line arguments
Parser = argparse.ArgumentParser()
Parser.add_argument('--Path', default='Data_set/car/frame')
Parser.add_argument('--ModelType', default='car')
Args = Parser.parse_args()
Path = Args.Path
typeName = Args.ModelType

# Initialization data path
frame_increase = 0
frameStart = dict({"car":20, "human":140, "vase":19})
frame_num = str(frameStart[typeName] + frame_increase).zfill(4) + ".jpg"
cap = cv2.VideoCapture(Path + frame_num)
success, image = cap.read()

# Initialize fine-tuned parametes for InverseLK . "Model":[learning rate, iteration]
finePara = dict({"car":[1, 250], "human":[0.75, 1000], "vase":[2, 2000]})
parameters = finePara[typeName]

# Initialize fine-tuned reference point
finePts = dict({"car" : [[161, 110], [299, 275]], "human" : [(258, 297), (282, 360)], "vase" : [[113, 81], [178, 154]]})
refPt = finePts[typeName]

# Start Tracking
p = np.zeros(6)
while success:
    clone = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop and equalize the image as template
    if frame_increase == 0:
        cv2.namedWindow('template', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('template', 320*5, 240*5)
        template = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        template = cv2.equalizeHist(template)
        cv2.imshow('template', template)
        print("top_left(x,y) and bottom_right(x,y) is")
        print(refPt)
    
    # Use InverseLk to track
    p = InverseLK(clone, template, parameters, refPt, p, frame_increase)

    warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    warp_mat = cv2.invertAffineTransform(warp_mat)
    refPt_other = [[refPt[1][0], refPt[0][1]], [refPt[0][0], refPt[1][1]]]
    newRefPt = np.hstack((refPt, [[1], [1]]))
    newRefPt_other = np.hstack((refPt_other, [[1], [1]]))
    newRefPt = np.dot(warp_mat, newRefPt.T).astype(np.int32)
    newRefPt_other = np.dot(warp_mat, newRefPt_other.T).astype(np.int32)
    print('new refpt', tuple(newRefPt.T[0]), tuple(newRefPt.T[1]))
    pts = np.array([newRefPt.T[0], newRefPt_other.T[0], newRefPt.T[1], newRefPt_other.T[1]])
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True,(255, 0, 0), 2)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 320*5, 240*5)
    cv2.imshow("image", image)
    cv2.waitKey(1)
        
    # Load the next frame    
    frame_increase += 1
    frame_num = str(frameStart[typeName] + frame_increase).zfill(4) + ".jpg"
    cap = cv2.VideoCapture(Path + frame_num)
    success, image = cap.read()

cap.release()
cv2.destroyAllWindows()
