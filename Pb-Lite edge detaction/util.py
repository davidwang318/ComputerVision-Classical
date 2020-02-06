import numpy as np
import cv2

def scaleImg(img_): 
	l = np.min(img_)
	h = np.max(img_)
	print(l, h)
	img = 255 * (img_- l) / (h-l)
	return img

def conv2d(img, kernel):
	kernel = np.flip(kernel)
	r, _ = kernel.shape
	img = cv2.copyMakeBorder(img, r-1, r-1, r-1, r-1, cv2.BORDER_CONSTANT, value=0)
	y, x = img.shape
	y_r = y - r + 1
	x_r  = x - r + 1
	convImg = np.zeros((y_r, x_r))
	for i in range(y_r):
		for j in range(x_r):
			convImg[i][j] = np.sum(img[i:i+r, j:j+r]*kernel)
	return convImg

def rotate(img, deg):
	h, w = img.shape
	R = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
	return cv2.warpAffine(img, R, (w, h))

def gaussian1d(x, sigma, order):
	exp1 = np.exp(-(x**2/(2.0*sigma**2)))
	if order == 0:
		return (1/np.sqrt(2*np.pi*sigma**2)) * exp1
	elif order == 1:
		return  -(x/(sigma**3*np.sqrt(np.pi*2))) * exp1
	elif order == 2:
		return -(sigma**2-x**2) / (sigma**5*np.sqrt(np.pi*2)) * exp1
	print("error in gaussian1d function!")

def gaussian2d(size, sigma, order):
	# size: kernel size, order: 0 for gaussian. else for log
	r = (size-1) / 2
	G = np.zeros((size, size))
	for x in range(-r, r+1):
		for y in range(-r, r+1):
			exp2 = np.exp(-(x**2+y**2) / (2.0*sigma**2))
			if order == 0: 
				G[x+r][y+r] = (1/(2.0*np.pi*sigma**2)) * exp2
			else:
				G[x+r][y+r] = -(1/(np.pi*sigma**4)) * (1-((x**2+y**2) / (2.0*sigma**2))) * exp2
	return G

def elongateGauss(size, factor, sigma, xOrder, yOrder):
	r = (size-1) / 2
	G = np.zeros((size, size))
	for x in range(-r, r+1):
		gx = gaussian1d(x, sigma*factor, xOrder)
		for y in range(-r, r+1):
			gy = gaussian1d(y, sigma, yOrder)
			G[x+r][y+r] = gx * gy
	return G

def gabor(sigma, rad, lamda, std):
	r = max(abs(std*sigma), abs(std*sigma))
	(y, x) = np.meshgrid(np.arange(-r, r+1), np.arange(-r, r+1))
	
	X = x*np.cos(rad) + y*np.sin(rad)
	Y = -x*np.sin(rad) + y*np.cos(rad) 
	gab = np.exp(-(X**2+Y**2)/(2.0*sigma**2)) * np.cos(2.0*np.pi*X/lamda)
	return gab


def halfMask(size, deg):
	mask = np.zeros((size, size))
	r = (size-1)/2
	axes = (r+1, r+1)
	center = (r, r)
	cv2.ellipse(mask, center, axes, deg, 0, 180, 1, -1)
	return mask