import numpy as np
import cv2
import pandas as pd

def readMatches(filePath):
	for i in range(1,6):
		fileName = "matching" + str(i) + ".txt"
		matchPts = pd.read_csv(filePath+fileName)
		drop = matchPts.drop_duplicates(keep = "last")
		drop.to_csv(filePath + "drop_" + fileName, sep='\t', index=False)

	correspondDicts = {'12' : [], '13' : [], '14' : [], '15' : [], '16' : [],
	                              '23' : [], '24' : [], '25' : [], '26' : [],
	                                         '34' : [], '35' : [], '36' : [],
	                                                    '45' : [], '46' : [],
	                                                               '56' : []}
	for imgIdx in range(1,6):
	    with open("Data/drop_matching"+str(imgIdx)+".txt","r") as f:
	    	for i, data in enumerate(f.readlines()):
	    		if i == 0: continue
	    		data = np.fromstring(data, dtype=float, sep=' ')
	    		refPt = [data[4], data[5]]
	    		for j in range(6, len(data), 3):
	    			dstPt = [data[j+1], data[j+2]]
	    			correspondDicts[str(imgIdx)+str(int(data[j]))].append([refPt, dstPt])
	for key in correspondDicts:
		correspondDicts[key] = np.array(correspondDicts[key], np.float32)

	return correspondDicts



def drawMatching(img1, img2, matchPts):
    r1, c1, _ = img1.shape
    r2, c2, _ = img2.shape
    imgMatch = np.zeros((max(r1, r2), c1+c2, 3)).astype(np.uint8)
    imgMatch[:r1, :c1] = img1[:, :]
    imgMatch[:r2, c1:] = img2[:, :]
    for pt1, pt2 in matchPts.astype(np.int64):
        x1, y1 = pt1
        x2, y2 = pt2
        x2 = x2 + c1
        imgMatch = cv2.line(imgMatch, (x1,y1), (x2,y2), (0, 255, 0), 1)
    return imgMatch


if __name__ == "__main__":
	filePath = "../Data/Imgs/"

	img1 = cv2.imread(filePath+'1.jpg')
	img2 = cv2.imread(filePath+'2.jpg')

	corDict = readMatches(filePath)
	imgMatch = drawMatching(img1, img2, corDict["12"])

	cv2.imshow("img1", img1)
	cv2.imshow("img2", img2)
	cv2.imshow("Match", imgMatch)

	cv2.waitKey()
	cv2.destroyAllWindows()