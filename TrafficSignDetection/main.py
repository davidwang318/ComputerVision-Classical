import cv2
import numpy as np
from hog import *
from svm import *


# SVM for label
trainingData, trainingLabels = loadData('Training')
testingData, testingLabels = loadData('Testing')
multiclassData = np.concatenate([trainingData, testingData], axis=0)
multiclassLabels = np.concatenate([trainingLabels, testingLabels], axis=0)
svm = createSVM(multiclassData, multiclassLabels)

# SVM for BLUE background noise
trainingData_, trainingLabels_ = loadData('Training', cropping=False)
testingData_, testingLabels_ = loadData('Testing', cropping=False)

negData = loadNegData('Negative_blue')

exemplarData = np.concatenate([trainingData,
                               testingData,
                               trainingData_,
                               testingData_,
                               negData], axis=0)
exemplarLabels = np.concatenate([trainingLabels,
                                 testingLabels,
                                 trainingLabels_,
                                 testingLabels_,
                                 -np.ones(negData.shape[0], dtype=int)], axis=0)

svm_blue = createExemplarSVM(exemplarData, exemplarLabels)

# SVM for RED background noise
negData = loadNegData('Negative_red')

exemplarData = np.concatenate([trainingData,
                               testingData,
                               trainingData_,
                               testingData_,
                               negData], axis=0)
exemplarLabels = np.concatenate([trainingLabels,
                                 testingLabels,
                                 trainingLabels_,
                                 testingLabels_,
                                 -np.ones(negData.shape[0], dtype=int)], axis=0)

svm_red = createExemplarSVM(exemplarData, exemplarLabels)

svm_color= [svm_red, svm_blue]

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 25.0, (1628, 1236))
labelset = [[21, 17, 1, 14, 19], [45, 38, 35]]
index = 0
while index <= 2860:
    img_temp = cv2.imread("frame_origin/frame%d.jpg" %index)
    img_r = cv2.imread("process_2_strong_red/frame%d.jpg" %index)
    img_b = cv2.imread("blue_hsv_mser/frame%d.jpg" %index)
    img_c = [img_r, img_b]
    temp_gray = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY)

    for order in range(2):
        # Contour threshold
        gray = cv2.cvtColor(img_c[order], cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255,cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            l = max(w, h)
            if order == 0: area_l, area_u, aspect_l, aspect_u = 500, 15000, 0.7, 1.3
            else: area_l, area_u, aspect_l, aspect_u = 300, 15000, 0.6, 1.4

            if area_l < w*h < area_u and aspect_l <= w/h <= aspect_u:
                crop = temp_gray[y:y+h, x:x+w]
                crop = cv2.resize(crop, (64, 64))
                hist = np.array([computeHOG(crop)]).astype(np.float32)
                confidence = predictExemplarSVM(svm_color[order], hist)[0]
                if confidence:
                    label = predictSVM(svm, hist)[0]
                    if label in labelset[order]:
                        sign = cv2.imread("CleanSign/%d.png" %label)
                        sign = cv2.resize(sign, (l,l))
                        tlx, tly = int(x + w/2 -l/2), int(y + h/2 - l/2)
                        # cv2.rectangle(img_c[order], (x, y), (x+w, y+h), (0, 255, 0), 2)
                        # cv2.rectangle(img_temp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.rectangle(img_c[order], (tlx, tly), (tlx + l, tly + l), (0, 255, 2))
                        cv2.rectangle(img_temp, (tlx, tly), (tlx + l, tly + l), (0, 255, 0), 2)
                        if tlx < l:
                            img_temp[tly: tly+l, tlx+l: tlx+2*l] = sign
                        else:
                            img_temp[tly: tly+l, tlx-l: tlx] = sign 
                    else:
                        cv2.rectangle(img_c[order], (x, y), (x+w, y+h), (255, 0, 0), 2)
                        #cv2.rectangle(img_temp, (x, y), (x+w, y+h), (255, 0, 0), 2)

                else:
                    cv2.rectangle(img_c[order], (x, y), (x+w, y+h), (0, 0, 255), 2)  
                    #cv2.rectangle(img_temp, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # out.write(img_temp)  
    # cv2.imwrite("Result_2/frame%d.jpg" %index, img_temp)

    cv2.imshow("Template", img_temp)
    # cv2.imshow("Blue channel", img_b)
    # cv2.imshow("Red channel", img_r)
    cv2.waitKey(1)
    index += 1
# out.release()
cv2.destroyAllWindows()
