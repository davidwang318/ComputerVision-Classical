import cv2
import numpy as np

# Our implementation of HOG
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    for i in range(0, int(img.shape[0]/celly)):
        for j in range(0,int(img.shape[1]/cellx)):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hists = np.array(hists)
    print(hists.shape)
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= np.linalg.norm(hist) + eps

    return hist

# The one we used in this project
def computeHOG(img):
    '''

    :param img: 64*64 gray image
    :return: HOG descriptor with shape (1024,)
    '''
    hog = cv2.HOGDescriptor((64, 64), (64, 64), (8, 8), (8, 8), 16)
    descriptor = hog.compute(img).reshape(-1)

    return descriptor


def crop_img(_img):
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _, contours, hier = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    crop_imgs = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 750 < w*h < 6000 and 0.7 <= w/h <= 1.3:
            crop_img = _img[y:y + h, x:x + w]
            crop_imgs.append(crop_img)
    crop_imgs = np.array(crop_imgs)

    return crop_imgs


if __name__ == "__main__":
    index = 76
    image = cv2.imread("./Blue_result/%d.jpg" %index)
    crops = crop_img(image)

    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        hog_descriptor = hog(gray)
        # hog_descriptors.append(hog_descriptor)
        # result = cvhog.compute(gray)
        result = computeHOG(gray)

    print(np.linalg.norm(hog_descriptor - result))

