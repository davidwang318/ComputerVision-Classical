import numpy as np
import cv2


def jacobian(x_shape, y_shape):
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y)
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)

    return jacob


def InverseLK(img, tmp, parameters, rect, p, count):

    # Initialization
    rows, cols = tmp.shape
    lr, iteration = parameters
    threshold = 0.005  # threshold for delta_p

    # Calculate gradient of template
    grad_x = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate Jacobian
    jacob = jacobian(cols, rows)

    # Set gradient of a pixel into 1 by 2 vector
    grad = np.stack((grad_x, grad_y), axis=2)
    grad = np.expand_dims((grad), axis=2)
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))
         
    # Compute Hessian matrix
    hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
 
    # while error_mean > threshold:
    for _ in range(iteration):
        # Calculate warp image
        warp_mat = np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
        warp_img = cv2.warpAffine(img, warp_mat, (0, 0))[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        
        # Equalize the image
        warp_img = cv2.equalizeHist(warp_img)

        # Compute the error term
        error = tmp.astype(float) - warp_img.astype(float)
     
        # Compute steepest-gradient-descent update
        error = error.reshape((rows, cols, 1, 1))
        update = (steepest_descents_trans * error).sum((0,1))
        d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
        
        #Update p
        d_p_deno = (1+d_p[0]) * (1+d_p[3])- d_p[1]*d_p[2]
        d_p_0 = (-d_p[0] - d_p[0]*d_p[3] + d_p[1]*d_p[2]) / d_p_deno 
        d_p_1 = (-d_p[1]) / d_p_deno
        d_p_2 = (-d_p[2]) / d_p_deno
        d_p_3 = (-d_p[3] - d_p[0]*d_p[3] + d_p[1]*d_p[2]) / d_p_deno
        d_p_4 = (-d_p[4] - d_p[3]*d_p[4] + d_p[2]*d_p[5]) / d_p_deno
        d_p_5 = (-d_p[5] - d_p[0]*d_p[5] + d_p[1]*d_p[4]) / d_p_deno

        p[0] += lr * (d_p_0 + p[0]*d_p_0 + p[2]*d_p_1)
        p[1] += lr * (d_p_1 + p[1]*d_p_0 + p[3]*d_p_1)
        p[2] += lr * (d_p_2 + p[0]*d_p_2 + p[2]*d_p_3)
        p[3] += lr * (d_p_3 + p[1]*d_p_2 + p[3]*d_p_3)
        p[4] += lr * (d_p_4 + p[0]*d_p_4 + p[2]*d_p_5)
        p[5] += lr * (d_p_5 + p[1]*d_p_4 + p[3]*d_p_5)
      
    cv2.imshow('equalize_img', warp_img)
    return p

