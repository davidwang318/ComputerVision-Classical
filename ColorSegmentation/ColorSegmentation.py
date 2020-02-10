import cv2
import numpy as np
import os
import em


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def gmm_train(address, k, iteration):
    images = load_images_from_folder(address)

    hist_bgr = np.zeros((256, 256, 256))
    for img in images:
        bgr_update = cv2.calcHist([images[0]], [0, 1, 2], None, [256, 256, 256], [50, 256, 50, 256, 50, 256])
        hist_bgr += bgr_update
    hist_bgr /= len(images)
    hist_bgr /= hist_bgr.sum()

    num_data = 2
    train = np.zeros((1, 3))  # Initialize training data
    for scale in range(1, num_data):
        scale /= num_data
        scale *= hist_bgr.max()
        train_tmp = np.stack(np.where(hist_bgr >= scale), axis=1)
        if scale == 1 / num_data:
            train = train_tmp
        else:
            train = np.concatenate((train, train_tmp))
    train /= 255  # Training data normalization
    gmmModel = em.GaussianMixture(train, k)

    for i in range(iteration):
        print(address+" training: ", i)
        gmmModel.train()
    return gmmModel


def gmm_fit(img, k1, model, threshold1, threshold2):
    img_shape = np.shape(img)
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    tmp = np.reshape(blur, (-1, 3)) / 255  # Testing data normalization

    if k1:
        prob = em.gaussian(tmp, model.getModel()[0][k1], model.getModel()[1][k1]).reshape((-1))
        prob = np.reshape(prob, (img_shape[0], img_shape[1]))

    else:
        prob = model.getPdf(tmp)
        prob = np.reshape(prob, (img_shape[0], img_shape[1]))

    _, img_thresh = cv2.threshold(prob, threshold1, 255, cv2.THRESH_BINARY)
    _, img_thresh_slack = cv2.threshold(prob, threshold2, 255, cv2.THRESH_BINARY)
    img_thresh = img_thresh.astype(np.uint8)
    seg = cv2.bitwise_and(img, img, mask=img_thresh)  # segmentaion
    return seg, img_shape, img_thresh, img_thresh_slack


# Training Model
yellow_gmm = gmm_train('yellow', 1, 50)
green_gmm = gmm_train('green', 1, 50)
orange_gmm = gmm_train('orange', 4, 200)

# Video Reader
vid = cv2.VideoCapture("../detectbuoy.avi")
success, img = vid.read()
count = 0

# # Video Writer
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('Project3_result.avi', fourcc, 20.0, (640, 480))

while success:

    img_result = img.copy()

    # Fitting image
    y_seg, y_img_shape, y_img_thresh, y_img_thresh_slack = gmm_fit(img, None, yellow_gmm, 1, 10e-3)
    g_seg, g_img_shape, g_img_thresh, g_img_thresh_slack = gmm_fit(img, None, green_gmm, 20, 10)
    o_seg, o_img_shape, o_img_thresh, o_img_thresh_slack = gmm_fit(img, 2, orange_gmm, 10e-15, 10e-50)
    seg = [y_seg, g_seg, o_seg]
    img_shape = [y_img_shape, g_img_shape, o_img_shape]
    img_thresh = [y_img_thresh, g_img_thresh, o_img_thresh]
    img_thresh_slack = [y_img_thresh_slack, g_img_thresh_slack, o_img_thresh_slack]

    #  Find bounding box
    color = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]
    img_total = [img.copy(), img.copy(), img.copy()]
    img_show = []
    for i in range(len(seg)):
        _, contours, _ = cv2.findContours(img_thresh[i], 1, 2) # opencv3: return contours, cnts, hierarchy.
        x = np.shape(img)[0]-1
        y = np.shape(img)[1]-1
        w = h = 0
        if contours:
            area = 100
            for cnt in contours:
                if cv2.contourArea(cnt) > area:
                    area = cv2.contourArea(cnt)
                    x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(seg[i], (x-int(0.5*w), y-int(0.5*h)), (x + int(1.5*w), y + int(1.5*h)), (0, 255, 0), 2)

        mask = np.zeros((img_shape[i][0], img_shape[i][1])).astype(np.uint8)
        cv2.rectangle(mask, (x - int(0.5 * w), y - int(1.5*h)), (x + int(1.5 * w), y + int(1.5 * h)), 255, -1)
        mask = cv2.bitwise_and(mask, img_thresh_slack[i].astype(np.uint8))
        img_show.append(cv2.bitwise_and(img_total[i], img_total[i], mask=mask))

        # Fit circle
        _, contours, _= cv2.findContours(mask, 1, 2)
        if contours:
            area = 0
            center = (0, 0)
            radius = 0
            for cnt in contours:
                if cv2.contourArea(cnt) > area:
                    area = cv2.contourArea(cnt)
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
            cv2.circle(img_result, center, radius, color[i], 2)

    cv2.imshow("Result", img_result)
    # cv2.imwrite("result/frame%d.jpg" % count, img_result)
    # cv2.imshow("y_channel", img_show[0])
    # cv2.imwrite("y_channel/frame%d.jpg" % count, img_show[0])
    # cv2.imshow("g_channel", img_show[1])
    # cv2.imwrite("g_channel/frame%d.jpg" % count, img_show[1])
    # cv2.imshow("o_channel", img_show[2])
    # cv2.imwrite("o_channel/frame%d.jpg" % count, img_show[2])
    count += 1
    cv2.waitKey(1)

    success, img = vid.read()

vid.release()
cv2.destroyAllWindows()
