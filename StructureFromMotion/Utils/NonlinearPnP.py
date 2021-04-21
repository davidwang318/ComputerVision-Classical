from PnPRANSAC import *
from scipy.spatial.transform import Rotation


def ReprojectionErrorPnP(projPara, worldPts, imgPts):
    # Restore projection matrix
    R, _ = cv2.Rodrigues(projPara[:3][:, None])
    T = np.array(projPara[3:]).reshape((3,1))
    proj = np.concatenate((R,T), axis=1)
    # Construct the Camera matrix
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])
    proj = np.matmul(K, proj)
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
    # reprojError = np.sum(np.linalg.norm(diff, axis=1)**2)
    reprojError = np.sum(np.linalg.norm(diff, axis=1))

    return reprojError

def NonlinearPnP(worldPts, imgPts, camMatrix, K):
    print("*** NonlinearPnP process:")
    # Restore the projection matrix from the camera matrix
    projMatrix = np.matmul(np.linalg.inv(K), camMatrix)
    R = projMatrix[:, :3]
    T = projMatrix[:, 3]
    R /= np.linalg.norm(R[:,0])
    T /= np.linalg.norm(R[:,0])
    print(len(worldPts))

    s, _, vt = np.linalg.svd(R)
    R_refined = np.matmul(s, vt)
    if np.linalg.det(R_refined) < 0:
    	R_refined = -R_refined
    	T = -T

    # Transform into quaternion
    R_quat, _ = cv2.Rodrigues(R_refined)
    projPara = np.concatenate((R_quat.flatten(), T.flatten()))

    # Nonlinear Triangulation
    print("Original Mean ReprojectionErrorPnP: ", ReprojectionErrorPnP(projPara, worldPts, imgPts) / len(worldPts))
    print("Optimizing......")
    nonlinearRes = optimize.minimize(ReprojectionErrorPnP, projPara, method='Nelder-Mead', args=(worldPts, imgPts))
    print("Optimized Mean ReprojectionErrorPnP: ", ReprojectionErrorPnP(nonlinearRes.x, worldPts, imgPts) / len(worldPts))

    # Recover the rotation and translation
    R_best, _ = cv2.Rodrigues(nonlinearRes.x[:3][:, None])
    T_best = nonlinearRes.x[3:]
    P_best = np.concatenate((R_best, T_best.reshape((3,1))), axis=1)

    return P_best

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
    # print(best_point[0:5])
    # print(best_inlier[0:5])
    worldPts_inlier, imgPts_inlier, P_best_ransac = PnPRANSAC(best_point, best_inlier[:, 1, :], 20000, 1, K)
    P_best_refined = NonlinearPnP(worldPts_inlier, imgPts_inlier, P_best_ransac, K)

    # Plot the base triangle
    TrianglePts = np.array([[0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [-1, 0, 1, 1],
                            [0, 0, 0, 1]])
    plt.plot(TrianglePts[:,0], TrianglePts[:,2], color='black')

    # Plot the pose triangle
    P_best_ransac = np.matmul(np.linalg.inv(K), P_best_ransac)
    R = P_best_ransac[:,:3]
    T = P_best_ransac[:, 3]

    u, _, vt = np.linalg.svd(R)
    R_true = np.matmul(u, vt)
    if np.linalg.det(R_true) < 0:
        R = -R
        T = -T

    Pts = TrianglePts
    proj = np.concatenate([np.linalg.inv(R), -np.matmul(np.linalg.inv(R), T[:, None])], axis=1).T
    Pts = np.matmul(Pts, proj)
    # Pts = np.matmul(Pts, P_pnp.T)
    plt.plot(Pts[:,0], Pts[:,2], color='red')

    # Plot the pose triangle
    R = P_best_refined[:, :3]
    T = P_best_refined[:, 3]
    Pts = TrianglePts
    proj = np.concatenate([R.T, -np.matmul(R.T, T[:, None])], axis=1).T
    Pts = np.matmul(Pts, proj)
    # Pts = np.matmul(Pts, P_pnp.T)
    plt.plot(Pts[:,0], Pts[:,2], color='green')

    # Plot the original3d points
    plt.scatter(best_point[:,0], best_point[:,2], color="red", marker='.')

    plt.axis('equal')
    plt.show()