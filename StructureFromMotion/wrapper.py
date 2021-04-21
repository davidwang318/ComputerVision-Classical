from Utils.NonlinearPnP import *
from Utils.BundleAdjustment import *
from mpl_toolkits.mplot3d import Axes3D


def drawMatchingRANSAC(curDict, matchPts_inlier):
    imgL = cv2.imread(filePath+'1.jpg')
    imgR = cv2.imread(filePath+'2.jpg')
    imgMatch = drawMatching(imgL, imgR, curDict, matchPts_inlier)
    cv2.imshow("Match12", imgMatch)
    # cv2.imwrite(savePath + 'Match12.jpg', imgMatch)
    cv2.waitKey()
    cv2.destroyAllWindows()


def intersect2d(X, Y):
    X = np.tile(X[:,:,None], (1, 1, Y.shape[0]) )
    Y = np.swapaxes(Y[:,:,None], 0, 2)
    Y = np.tile(Y, (X.shape[0], 1, 1))
    eq = np.all(np.equal(X, Y), axis = 1)
    eq = np.any(eq, axis = 1)
    return np.nonzero(eq)[0]

def plotResult(proj_set, worldPts):
    # Plot the base triangle
    TrianglePts = np.array([[0, 0, 0, 1],
                            [1, 0, 1, 1],
                            [-1, 0, 1, 1],
                            [0, 0, 0, 1]])

    # Plot the pose triangle
    color_set = ["black", "red", "green", "blue", "brown", "crimson"]
    for i, pose in enumerate(proj_set):
        Pts = TrianglePts
        R = pose[:,:3]
        T = pose[:, 3]
        proj = np.concatenate([R.T, -np.matmul(R.T, T[:, None])], axis=1).T
        Pts = np.matmul(Pts, proj)
        plt.plot(Pts[:,0], Pts[:,2], color=color_set[i])

    # Plot the original3d points
    plt.scatter(worldPts[:,0], worldPts[:,2], color="blue", marker='.')

    plt.axis('equal')
    plt.show()

def plot3d(worldPts):
    x = worldPts[:, 0]
    y = worldPts[:, 1]
    z = worldPts[:, 2]
    in_list = np.logical_and(z < 100, z > -10)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[in_list], z[in_list], -y[in_list], marker='.', color='blue')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')
    plt.axis('equal')
    plt.show()



if __name__ == '__main__':

    # Read / Organize the correspondence
    filePath = "Data/"
    corDict = readMatches(filePath)

    # Preprocess the matches:
    process_set = ['12', '13', '23', '14', '24', '34', '15', '25', '35', '45', '16', '26', '36', '46', '56']
    F_set, matchPts_inlier_set = [], []
    for i, m in enumerate(process_set):
        if i == 0: 
            iteration = 50000
            threshold = 0.0025
        else: 
            iteration = 10000
            threshold = 0.005
        print("processing dict: ", m)
        tmpDict = corDict[m]
        if len(tmpDict) == 0: continue
        F_tmp, matchPts_inlier_tmp = GetInlierRANSAC(tmpDict, iteration, threshold)
        F_set.append(F_tmp)
        matchPts_inlier_set.append(matchPts_inlier_tmp)
        corDict[m] = matchPts_inlier_tmp[..., :2]



    # Use the first two image to find initial pose and 3D points
    curDict = corDict['12']
    F = F_set[0]
    matchPts_inlier = matchPts_inlier_set[0]
    # drawMatchingRANSAC(curDict, matchPts_inlier) # Draw the inlier matcjes

    # Initialize the Intrinsic Parameters
    K = np.array([[568.996140852, 0, 643.21055941],
                  [0, 568.988362396, 477.982801038],
                  [0, 0, 1]])

    # Restore the Essential matrix
    E = EssentialMatrixFromFundamentalMatrix(F, K)

    # Extract the poses 
    poses = ExtractCameraPose(E)

    # Linear triangulation: worldPts_tri: 3d world point(n*3), imgPts_tri: 2d homo image point(n*2*3)
    pose_tri, worldPts_tri, imgPts_tri = DisambiguateCameraPose(poses, matchPts_inlier, K) 
    PlotTriangulation([pose_tri], [worldPts_tri])

    # # Nonlinear triangulation
    # worldPts_tri = NonlinearTriangulation(worldPts_tri, imgPts_tri, pose_tri, K)
    # PlotComparison(pose_tri, worldPts_tri, worldPts_non_tri)

    worldPts_inlier, imgPts_inlier, P_best_ransac, _ = PnPRANSAC(worldPts_tri, imgPts_tri[:, 1, :], 20000, 1, K)
    P_best_refined = NonlinearPnP(worldPts_inlier, imgPts_inlier, P_best_ransac, K)

    #########################################################################################################
    pose_world = (np.zeros(3), np.identity(3))
    proj_world = (np.concatenate([pose_world[1], pose_world[0][:, None]], axis=1))
    proj_set = []
    proj_set.append(proj_world)
    proj_set.append(P_best_refined)
    proj_set = np.array(proj_set)

    # world point set / image point set / camera_idx / worldPts_idx
    worldPts_set = worldPts_tri

    imgPts_set = np.round_(np.concatenate((imgPts_tri[:,0,:2], imgPts_tri[:,1,:2]), axis=0), decimals=2)

    camera_idx_set = np.zeros(len(imgPts_set)).astype(np.int32)
    camera_idx_set[:len(imgPts_tri)] = 0 # 1.jpg
    camera_idx_set[len(imgPts_tri):] = 1 # 2.jpg

    worldPts_idx = np.concatenate((np.arange(len(imgPts_tri)), np.arange(len(imgPts_tri))))

    plotResult(proj_set, worldPts_set)
    plot3d(worldPts_tri)

    # Find the correspondence between worldPts and image point in camera i
    for dstCam_idx in range(3, 7):
        searchPts_set = None
        searchCamera_set = None

        # Find process_imgPts
        for i in range(1, dstCam_idx): #13, 23
            searchDict = corDict[str(i)+str(dstCam_idx)] # n*2*2
            if(len(searchDict) == 0): continue
            if searchPts_set is None:
                searchPts_set = searchDict # 1 & 2 points
                searchCamera_set = np.ones(len(searchDict)) * (i-1)
            else:
                searchPts_set = np.concatenate((searchPts_set, searchDict), axis=0)
                searchCamera_set = np.concatenate((searchCamera_set, np.ones(len(searchDict)) * (i-1)), axis=0)

        common_list = intersect2d(np.round_(searchPts_set[:,0,:].astype(np.float32), decimals=2), np.round_(imgPts_set.astype(np.float32), decimals=2))
        print("Current 3D points: ", len(worldPts_set))
        print("Process 2D points: ", len(common_list))
        corr_imgPts = searchPts_set[common_list, 0, :]
        process_imgPts = searchPts_set[common_list, 1, :]
        homoAxis = np.ones((len(process_imgPts), 1))
        process_imgPts = np.concatenate((process_imgPts, homoAxis), axis=1)
        
        # Find process_worldPts_idx / process_camera_idx
        process_worldPts_idx, process_camera_idx = [], []
        for imgPt in corr_imgPts:
            common_list2 =  np.logical_and(np.round_(imgPts_set[:, 0].astype(np.float32), decimals=2) == round(imgPt[0], 2), 
                                           np.round_(imgPts_set[:, 1].astype(np.float32), decimals=2) == round(imgPt[1], 2))
            world_idx = np.where(common_list2)[0][0]
            camera_idx = np.where(common_list2)[0][0]
            process_worldPts_idx.append(world_idx)
            process_camera_idx.append(camera_idx)
        process_worldPts_idx = worldPts_idx[np.array(process_worldPts_idx)]
        process_camera_idx = camera_idx_set[np.array(process_camera_idx)]

        # Find process_worldPts
        process_worldPts = worldPts_set[process_worldPts_idx]
        print("Process 3D points: ", len(process_worldPts))

        # PnP to restore Pose
        worldPts_inlier, imgPts_inlier, P_best_ransac, inlier_list = PnPRANSAC(process_worldPts, process_imgPts, 50000, 1, K)
        P_best_refined = NonlinearPnP(worldPts_inlier, imgPts_inlier, P_best_ransac, K)

        # LinearTriangulation
        corr_imgPts_tri, process_imgPts_tri = searchPts_set[:, 0, :], searchPts_set[:, 1, :]
        worldPts_tri_new = LinearTriangulation2(corr_imgPts_tri, process_imgPts_tri, np.matmul(K, P_best_refined), proj_set, searchCamera_set.astype(np.int32), K)
        proj_set = np.concatenate((proj_set, P_best_refined[None, ...]), axis=0)
        plotResult(proj_set, worldPts_tri_new)

        """
        Organize the data and update:
        1. worldPts_set
        2. imgPts_set
        3. camera_idx
        4. worldPts_idx
        5. proj_set
        """

        inlier_list = worldPts_tri_new[..., 2] > 0
        worldPts_tri_new = worldPts_tri_new[inlier_list]
        corr_imgPts_tri = corr_imgPts_tri[inlier_list]
        process_imgPts_tri = process_imgPts_tri[inlier_list]
        for new_worldPt, new_imgPt, corr_imgPt in zip(worldPts_tri_new, process_imgPts_tri, corr_imgPts_tri):
            common_list3 = intersect2d(np.round_(imgPts_set.astype(np.float32), decimals=2), np.round_(corr_imgPt[None, ...].astype(np.float32), decimals=2))
            if len(common_list3) == 0:
                imgPts_set = np.concatenate((imgPts_set, new_imgPt[None, ...]), axis=0)
                worldPts_set = np.concatenate((worldPts_set, new_worldPt[None, ...]), axis=0)
                camera_idx_set = np.concatenate((camera_idx_set, np.array([dstCam_idx-1])))
                worldPts_idx = np.concatenate((worldPts_idx, np.array([len(worldPts_set)-1])))

        # Bundle adjustment def __init__(self, cameraArray, points3D, points2D, cameraIndices, point2DIndices, K):
        bundleAdjustmentObj = PySBA2(proj_set, worldPts_set, imgPts_set, camera_idx_set, worldPts_idx, K)

        # cameraPose, points3D = bundleAdjustmentObj.bundleAdjust()
        points3D = bundleAdjustmentObj.bundleAdjust()


        # plotResult(cameraPose, points3D)
        # proj_set = cameraPose
        worldPts_set = points3D
        plotResult(proj_set, points3D)
        plot3d(points3D)