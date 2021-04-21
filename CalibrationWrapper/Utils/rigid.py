import math
import numpy as np

# Input: expects 2xN matrix of points
# Returns R,t
# R = 2x2 rotation matrix
# t = 2x1 column vector

def rigid_transform_2D(A, B):
    assert len(A) == len(B)
    num_rows, num_cols = A.shape;

    if num_rows != 2:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 2:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # subtract mean
    Am = A - np.tile(centroid_A, (1, num_cols))
    Bm = B - np.tile(centroid_B, (1, num_cols))

    H = Am * np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    
    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B
    
    angle = math.atan2(R[1,0],R[0,0])

    return R, t, angle

if __name__ == '__main__':
    # Test with random data

    # Random rotation and translation
    angle = np.random.randn() * math.pi/2.
    c = math.cos(angle)
    s = math.sin(angle)
    R = np.mat([[c,-s],[s,c]])
    t = np.mat(np.random.rand(2,1))* 10

    # number of points
    n = 100

    A = np.mat(np.random.rand(2, n));
    A = A * 200

    B = R*A + np.tile(t, (1, n))
    
    #for i in range(B.shape[1]):
    #    B[0,i] = B[0,i] + np.random.randn()
    #    B[1,i] = B[1,i] + np.random.randn()

    # Recover R and t
    ret_R, ret_t, ret_angle = rigid_transform_2D(A, B)
    
    # Compare the recovered R and t with the original
    B2 = (ret_R*A) + np.tile(ret_t, (1, n))

    # Display result
    if False:
        import cv2
        ima = np.zeros([800,800,3],np.uint8)
        for i in range(A.shape[1]):
            pt = (int(round(A[0,i]))+400, int(round(A[1,i]))+400)
            cv2.circle(ima, pt, 3, (255,255,0) )
            pt = (int(round(B[0,i]))+400, int(round(B[1,i]))+400)
            cv2.circle(ima, pt, 3, (0,255,0) )
            pt = (int(round(B2[0,i]))+400, int(round(B2[1,i]))+400)
            cv2.circle(ima, pt, 5, (0,0,255), 2 )
        cv2.imshow("Points", ima)
        cv2.waitKey(0)

    # Find the root mean squared error
    err = B2 - B
    err = np.multiply(err, err)
    err = np.sum(err)
    rmse = math.sqrt(err/n);

    print("Points A")
    print(A)
    print("")

    print("Points B")
    print(B)
    print("")

    print("Ground truth rotation")
    print(R)

    print("Recovered rotation")
    print(ret_R)
    print("")

    print("Ground truth translation")
    print(t)

    print("Recovered translation")
    print(ret_t)
    print("")
    
    print("Ground angle")
    print(angle)
    print("Recovered angle")
    print(ret_angle)
    print("")

    print("RMSE:", rmse)

    if rmse < 1e-5:
        print("Everything looks good!\n");
    else:
        print("Hmm something doesn't look right ...\n");
