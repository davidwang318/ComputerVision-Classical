#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code


Author(s): 
Lih-Narn Wang (ytcdavid@terpmail.umd.edu)
M.Eng in Robtics,
University of Maryland, College Park

Nitin J. Sanket (nitin@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park

Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def dogFilters(o, size, sigma):
	# o: orientation, s: number of sigma, size: gaussian kernel
	sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
	deg = np.linspace(0, 360, o, endpoint=False)
	bank = []
	for sig in sigma:
		kernel = conv2d(gaussian2d(size, sig, 0), sobel)
		#kernel = convolve2d(gaussian2d(size, sig), sobel)
		for d in deg:
			kernelRot = rotate(kernel, d)
			bank.append(kernelRot)
	return bank

def lmFilters(sigma):
	'''
	banksize: 49x49x48
	scales:
		small: 2**0, 2**0.5, 2**1
		large: 2**0.5, 2**1, 2**1.5
	elogation factor = 3
	'''
	deg = np.linspace(0, 360, 6, endpoint=False)
	bank = []
	kernel = [[],[]] # 0 for 1 order, 1 for 2 order
	for i in range(3):
		sig = sigma[i]
		kernel[0].append(elongateGauss(49, 3, sig, 0, 1))
		kernel[1].append(elongateGauss(49, 3, sig, 0, 2))
	for i in range(3):
		for j in range(2):
			for d in deg:
				bank.append(rotate(kernel[j][i], d))
	for sig in sigma:
		bank.append(gaussian2d(49, sig, 1))
	for sig in sigma:
		bank.append(gaussian2d(49, sig*3, 1))
	for sig in sigma:
		bank.append(gaussian2d(49, sig, 0))
	return bank


def gaborFilters(sigma, o):
	bank = []
	deg = np.linspace(0, np.pi, o, endpoint=False)
	for sig in sigma:
		for i in range(1, 3):
			lamda = float(sig)/i
			for d in deg:
				bank.append(gabor(sig, d, lamda, 3))
	return bank

def maskFilters(size, o):
	bank = []
	deg = np.linspace(0, 180, o, endpoint = False)
	for size_ in size:
		for deg_ in deg:
			lMask = halfMask(size_, deg_)
			rMask = halfMask(size_, deg_+180)
			bank.append([lMask, rMask])
	return bank

def getTextonMap(img, filters, textonBin):
	textonVec = np.array([])
	imgTexton = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for kernel in filters:
		tmp = cv2.filter2D(imgTexton, -1, kernel)
		tmp = tmp.flatten()[..., np.newaxis]
		textonVec = np.concatenate((textonVec, tmp), axis=1) if textonVec.size else tmp
	kmeans_ = KMeans(n_clusters=textonBin)
	kmeans_.fit(textonVec)
	return np.reshape(kmeans_.labels_, (imgTexton.shape))

def getBrightMap(img, brightBins):
	imgBright = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	brightVec = imgBright.reshape((img.shape[0]*img.shape[1]), 1)
	kmeans_  = KMeans(n_clusters=brightBins)
	kmeans_.fit(brightVec)
	brightMap = np.reshape(kmeans_.labels_, imgBright.shape)
	return brightMap

def getColorMap(img, colorBins):
	imgColor = img.copy()
	colorVec = imgColor.reshape((img.shape[0]*img.shape[1]), 3)
	kmeans_ = KMeans(n_clusters=colorBins)
	kmeans_.fit(colorVec)
	colorMap = np.reshape(kmeans_.labels_, (img.shape[0], img.shape[1]))
	return colorMap

def gradMap(img, numBin, maskBank):
	numMask = len(maskBank)
	gradMap_ = np.zeros((img.shape[0], img.shape[1], numMask))
	for i in range(numMask):
		chiSqrDist = np.zeros(img.shape)
		tmp = np.zeros(img.shape)
		for j in range(numBin):
			binaryImg = (img==j).astype(np.float64)
			lImg = cv2.filter2D(binaryImg, -1, maskBank[i][0])
			rImg = cv2.filter2D(binaryImg, -1, maskBank[i][1])
			chiSqrDist = chiSqrDist + 0.5*(lImg-rImg)**2 / (lImg+rImg+0.1**5)
		gradMap_[:, :, i] = chiSqrDist
	return np.mean(gradMap_, axis=2)

def saveFilters(bank, path, name):
	cv2.imwrite(os.path.join(path , 'name'+str(i)+'.png'), img)
	return

def showFilters(bank, name, row, col, isMask, color=None):
	plt.figure(figsize=(32,8))
	plt.title('Filter Bank of '+name+' filters')
	if isMask:
		i = 1
		for f in bank:
			plt.subplot(row, col, i)
			plt.axis('off')
			plt.imshow(f[0], cmap=color)
			i = i+1
			plt.subplot(row, col, i)
			plt.axis('off')
			plt.imshow(f[1], cmap=color)
			i = i+1
	else:
		for i in range(len(bank)):
			plt.subplot(row, col, i+1)
			plt.axis('off')
			plt.imshow(bank[i], cmap=color)
	plt.tight_layout()
	plt.savefig('Result/Filters/'+name+'.png')
	# plt.show()
	return

def main():
	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	dogBank = dogFilters(o=16, size=11, sigma=[1, 2, 3, 4])
	print('DoG filter number: ', len(dogBank))
	showFilters(dogBank, 'DoG Bank', 4, len(dogBank)/4, False, 'gray')

	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	lmsSigma = np.array([1, 2**0.5, 2, 2**1.5])
	lmlSigma = np.array([2**0.5, 2, 2**1.5, 4])
	lmsBank = lmFilters(lmsSigma)
	lmlBank = lmFilters(lmlSigma)
	print('LMS filter number: ', len(lmsBank))
	print('LML filter number: ', len(lmlBank))
	showFilters(lmsBank, 'LMS Bank', 4, len(lmsBank)/4, False, 'gray')
	showFilters(lmlBank, 'LML Bank', 4, len(lmlBank)/4, False, 'gray')
	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""
	gaborBank = gaborFilters(sigma=[5, 10, 20], o=8)
	print('Gabor filter number: ', len(gaborBank))
	showFilters(gaborBank, 'Gabor Bank', 3, len(gaborBank)/3, False, 'gray')

	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	maskBank = maskFilters(size=[5, 11, 21], o=8)
	print('Mask filter number: ', len(maskBank), ', 2')
	showFilters(maskBank, 'Mask Bank', 3, len(maskBank)/3*2, True, 'gray')

	for i in range(1, 11):
		index = str(i)
		img = cv2.imread('../BSDS500/Images/'+index+'.jpg')
		imgHeight, imgWidth, _ = img.shape

		"""
		Generate Texton Map
		Filter image using oriented gaussian filter bank
		Generate texture ID's using K-means clustering
		Display texton map and save image as TextonMap_ImageName.png,
		use command "cv2.imwrite('...)"
		"""
		textonBin = 64
		textonMap = getTextonMap(img, dogBank, textonBin)
		plt.clf()
		plt.imshow(textonMap)
		plt.tight_layout()
		plt.savefig('Result/Map/TextonMap'+index+'.png')
		# plt.show()

		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName'+index+'.png,
		use command "cv2.imwrite(...)"
		"""
		textonGrad = gradMap(textonMap, textonBin, maskBank)
		plt.clf()
		plt.imshow(textonGrad)
		plt.tight_layout()
		plt.savefig('Result/Grad/Tg'+index+'.png')
		# plt.show()
		
		"""
		Generate Brightness Map
		Perform brightness binning 
		"""
		brightBins = 16
		brightMap = getBrightMap(img, brightBins)
		plt.clf()
		plt.imshow(scaleImg(brightMap), cmap='gray')
		plt.tight_layout()
		plt.savefig('Result/Map/BrightMap'+index+'.png')
		# plt.show()
		
		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName'+index+'.png,
		use command "cv2.imwrite(...)"
		"""
		brightGrad = gradMap(brightMap, brightBins, maskBank)
		plt.clf()
		plt.imshow(brightGrad)
		plt.tight_layout()
		plt.savefig('Result/Grad/Bg'+index+'.png')
		# plt.show()
		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		colorBin = 16
		colorMap = getColorMap(img, colorBin)
		plt.clf()
		plt.imshow(colorMap)
		plt.tight_layout()
		plt.savefig('Result/Map/ColorMap'+index+'.png')
		# plt.show()
		
		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName'+index+'.png,
		use command "cv2.imwrite(...)"
		"""
		colorGrad = gradMap(colorMap, colorBin, maskBank)
		plt.clf()
		plt.imshow(colorGrad)
		plt.tight_layout()
		plt.savefig('Result/Grad/Cg'+index+'.png')
		# plt.show()

		"""
		Read Sobel Baseline
		"""
		sobelBase = cv2.imread('../BSDS500/SobelBaseline/'+index+'.png', 0)

		"""
		Read Canny Baseline
		"""
		cannyBase = cv2.imread('../BSDS500/CannyBaseline/'+index+'.png', 0)

		"""
		Combine responses to get pb-lite output
		Display PbLite and save image as PbLite_ImageName.png
		use command "cv2.imwrite(...)"
		"""
		wSobel = 0.5
		wCanny = 1-wSobel
		PbLite = ((textonGrad+brightGrad+colorGrad)/3)*(wSobel*sobelBase+wCanny*cannyBase)
		plt.clf()
		plt.imshow(PbLite, cmap='gray')
		plt.tight_layout()
		plt.savefig('Result/PbLite'+index+'.png')
	#plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
 


