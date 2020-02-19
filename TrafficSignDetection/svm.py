import os
import cv2
import numpy as np
import csv
from hog import *


def readCSV(name):
    '''
    Read labels
    '''
    table = {}
    with open(name, 'r') as csvfile:
        label_csv = csv.reader(csvfile, delimiter=';')
        line_count = 0
        for row in label_csv:
            if line_count > 0:
                table[row[0]] = [int(i) for i in row[1:]]
            line_count += 1
    return table


def loadData(directory='Training', cropping=True):
    '''
    Load data
    '''
    labels = []
    dataset = []
    for subdir in os.listdir(directory):
        if subdir.endswith(".txt"):
            # Ignore the readme.txt file
            continue
        for filename in os.listdir(directory+'/'+subdir):
            # Read CSV files
            if not filename.endswith(".csv"):
                continue
            trainClass = readCSV(directory+'/'+subdir+'/'+filename)
            for data, label in trainClass.items():
                # Read images
                img = cv2.imread(directory+'/'+subdir+'/'+data)
                if cropping:
                    img = img[label[2]:label[4], label[3]:label[5]]
                resized = cv2.resize(img, (64, 64))
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

                # HOG descriptor
                labels.append(label[-1])
                dataset.append(computeHOG(gray))
                #cv2.imshow('img', crop)
                #cv2.waitKey(1)

    dataset = np.array(dataset, dtype=np.float32)
    labels = np.array(labels)
    return dataset, labels


def loadNegData(directory):
    '''
    Load negative/background data
    '''
    negData = []
    for sample in os.listdir(directory):
        img = cv2.imread(directory+'/'+sample, 0)
        #cv2.imshow('img', img)
        #cv2.waitKey(1)
        resized = cv2.resize(img, (64, 64))
        negData.append(computeHOG(resized))
    negData = np.array(negData, dtype=np.float32)

    return negData
    

def createSVM(trainingData, trainingLabels):
    '''
    Train multi-class SVM
    '''
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.train(trainingData, cv2.ml.ROW_SAMPLE, trainingLabels)
    
    return svm


def predictSVM(svm, feature, returnLabel=True):
    '''
    Predict sign class
    '''
    if returnLabel:
        return svm.predict(feature)[1]
    return svm.predict(feature, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1]


def createExemplarSVM(trainingData, trainingLabels, numClass=62):
    svm_ex = []
    # Given 60 classes, and 2 for hard-negative mining
    for c in range(62):
        ovaLabels = (trainingLabels==c)*2-1
        svm = createSVM(trainingData, ovaLabels)
        svm_ex.append(svm)
    return svm_ex


def predictExemplarSVM(svm_ex, feature, threshold=0.4):
    confidence = np.zeros(feature.shape[0], dtype=bool)

    for c in range(len(svm_ex)):
        # for c in [45, 21, 38, 35, 17, 1, 14, 19]:
        score = svm_ex[c].predict(feature, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1]
        prob = 1.0 / (1.0 + np.exp(score))
        if c in [45, 21, 38, 35, 17, 1, 14, 19]: 
            confidence |= prob.reshape(-1) > threshold
        else:
            # If high chance false, still True
            both = np.logical_not(confidence & (prob.reshape(-1) > 0.8))
            confidence &= both

    return confidence


if __name__ == "__main__":

    # Load necessary data for learning
    trainingData, trainingLabels = loadData('Training', cropping=False)
    testingData, testingLabels = loadData('Testing', cropping=False)
    #negData = loadNegData('Negative')
    #binData, binLabels = createBinData(trainingData, negData)


    # Train multi-class SVM
    svm = createSVM(trainingData, trainingLabels)

    # Predict sign class
    predict = predictSVM(svm, trainingData, True)
    print("Training accuracy: ", np.mean(trainingLabels == predict.reshape(-1)))

    predict = predictSVM(svm, testingData, True)
    print("Testing accuracy: ", np.mean(testingLabels == predict.reshape(-1)))

    # Train binary SVM
    svm_bin2 = createSVM(binData, binLabels)
    predict = predictSVM(svm_bin2, testingData, True)
    print('testing: ', np.mean(predict == 1))
    predict = predictSVM(svm_bin2, trainingData, True)
    print('training: ', np.mean(predict == 1))
    predict = predictSVM(svm_bin2, negData, True)
    print('backgroud: ', np.mean(predict == 1))

    # Train exemplar-SVM
    #exemplarData = np.concatenate([trainingData, testingData, negData], axis=0)
    #exemplarLabels = np.concatenate([trainingLabels, testingLabels, -np.ones(negData.shape[0], dtype=int)], axis=0)
    exemplarData = np.concatenate([trainingData, testingData], axis=0)
    exemplarLabels = np.concatenate([trainingLabels, testingLabels], axis=0)
    svm_ex = createExemplarSVM(exemplarData, exemplarLabels)
    #confidence = predictExemplarSVM(svm_ex, negData)
    #print("background: ", np.mean(confidence))
    confidence = predictExemplarSVM(svm_ex, trainingData)
    print("train: ", np.mean(confidence))
    confidence = predictExemplarSVM(svm_ex, testingData)
    print("test: ", np.mean(confidence))

    '''
    exemplarData = np.concatenate([trainingData, testingData, negData], axis=0)
    exemplarLabels = np.concatenate([trainingLabels, testingLabels, -np.ones(negData.shape[0], dtype=int)], axis=0)

    svm_ex = []
    for c in range(62):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
        ovaLabels = (exemplarLabels==c)*2-1
        svm.train(exemplarData, cv2.ml.ROW_SAMPLE, ovaLabels)
        svm_ex.append(svm)

    # Predict with exemplar-SVM
    confidence = np.zeros(negData.shape[0], dtype=bool)
    #confidence = np.zeros(trainingData.shape[0], dtype=bool)
    #confidence = np.zeros(testingData.shape[0], dtype=bool)
    for c in range(62):
        score = svm_ex[c].predict(negData, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1]
        # score = svm_ex[c].predict(trainingData, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1]
        # score = svm_ex[c].predict(testingData, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)[1]
        prob = 1.0 / (1.0 + np.exp(score))
        confidence |= prob.reshape(-1) > 0.4

    print('detection accuracy: ', np.mean(confidence))
    '''

    # Visualization
    cv2.waitKey(0)
    cv2.destroyAllWindows()
