from LinearPnP import *

def ReprojError(worldPts, imgPts, proj):
    # Construct the homogeneous 3d pts
    homoAxis = np.ones((len(worldPts), 1))
    worldPts_homo = np.concatenate((worldPts, homoAxis), axis=1)
    # Project the 3d pts to image coordinates
    projPts = np.matmul(worldPts_homo, proj.T)
    # Normalize the image pts
    projPts /= projPts[..., -1][..., None]
    # Remove the homogeneous axis
    projPts = projPts[..., :2]
    # Calculate the reprojection error
    diff = projPts - imgPts[:, :2]
    reprojError = np.linalg.norm(diff, axis=1)**2
    

    return reprojError

def PnPRANSAC(worldPts, imgPts, iteration, threshold, K):
    print("*** PNP RANSAC process:")
    # Initialize parameters used in RANSAC Process
    ptNum = len(worldPts)
    inNum, inList = 0, []
    bestP = None

    # Start the process
    while iteration > 0:
        # Generate six points
        indices = np.random.randint(ptNum, size=6)
        randWorldPts = worldPts[indices]
        randImgPts = imgPts[indices]
        # Calculate the pose
        P = linearPnP(randWorldPts, randImgPts, K)
        error = ReprojError(worldPts, imgPts, P)
        # Count inliers
        inList_tmp = error < threshold
        inNum_tmp = np.count_nonzero(inList_tmp)
        # Condition
        if inNum_tmp > inNum:
            inNum = inNum_tmp
            inList = inList_tmp
            bestP = P
            print("Current error: ", np.mean(error))
        iteration -= 1

    # Recalculate the result
    worldPts_inlier = worldPts[inList]
    imgPts_inlier = imgPts[inList]
    # P_best = linearPnP(worldPts_inlier, imgPts_inlier, K)
    P_best = bestP
    err = ReprojError(worldPts_inlier, imgPts_inlier, P_best)
    print("Last error: ", np.mean(err))
    print("Original Number: ", len(worldPts))
    print("Inlier Number: ", len(worldPts_inlier))


    return worldPts_inlier, imgPts_inlier, P_best, inList


if __name__ == '__main__':
    filePath = "../Data/Imgs/"
    savePath = "../Data/Imgs/GetInlierRANSAC/"
    corDict = readMatches(filePath)
    curDict = corDict['12']

    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    imgL = cv2.imread(filePath+str(1)+'.jpg')
    imgR = cv2.imread(filePath+str(2)+'.jpg')
    F, matchPts_inlier = GetInlierRANSAC(curDict, 50000, 0.0025)
    # imgMatch = drawMatching(imgL, imgR, curDict, matchPts_inlier)

    # cv2.imshow("Match12", imgMatch)
    # # cv2.imwrite(savePath + "Match"+str(i)+str(j) + '.jpg', imgMatch)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    E = EssentialMatrixFromFundamentalMatrix(F, K)
    poses = ExtractCameraPose(E)
    best_pose, best_point, best_inlier = DisambiguateCameraPose(poses, matchPts_inlier, K)
    # best_point: 3d world point(n*3), best_inlier: 2d homo image point(n*2*3)
    PlotTriangulation([best_pose], [best_point])

    # Organize the data
    _,_,P_best = PnPRANSAC(best_point, best_inlier[:, 1, :], 20000, 1, K)
    P_best = np.matmul(np.linalg.inv(K), P_best)

    # Plot the base triangle
    TrianglePts = np.array([[0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [-1, 0, 1, 1],
                            [0, 0, 0, 1]])
    plt.plot(TrianglePts[:,0], TrianglePts[:,2], color='black')

    # Plot the pose triangle
    R = P_best[:,:3]
    T = P_best[:, 3]

    u, _, vt = np.linalg.svd(R)
    R_true = np.matmul(u, vt)
    if np.linalg.det(R_true) < 0:
        print(np.linalg.det(R_true))
        R = -R
        T = -T

    Pts = TrianglePts
    proj = np.concatenate([np.linalg.inv(R), -np.matmul(np.linalg.inv(R), T[:, None])], axis=1).T
    Pts = np.matmul(Pts, proj)
    # Pts = np.matmul(Pts, P_pnp.T)
    plt.plot(Pts[:,0], Pts[:,2], color='red')

    # Plot the original3d points
    plt.scatter(best_point[:,0], best_point[:,2], color="red", marker='.')

    plt.axis('equal')
    plt.show()

