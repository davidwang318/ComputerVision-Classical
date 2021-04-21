from utils.findHomography import *
from utils.estimateCameraParameters import *


def reprojectError(mdPts, imgPts, K, P, mode='mean', distortion=[0, 0], returnPts=False):
    zeroAxis = np.zeros(54).reshape((54,1))
    homoAxis = np.ones(54).reshape((54,1))
    addAxis = np.concatenate((zeroAxis, homoAxis), axis=1)
    mdPts_homo = np.concatenate((mdPts, addAxis), axis=1)

    totalErr = 0
    dis_projPts_all = []
    for (imgPt, p) in zip(imgPts, P):
        p_ = np.matmul(K, p)
        projectPts = np.dot(mdPts_homo, p_.T)
        projectPts = projectPts / projectPts[:, 2][:, None]
        projectPts = projectPts[:, :2]
        dis_projPts = distortPts(projectPts, K, distortion)
        dis_projPts_all.append(dis_projPts)
        if mode == 'mean':
            err = np.mean(np.linalg.norm(dis_projPts - imgPt, axis=1))
        else:
            err = np.sum(np.linalg.norm(dis_projPts - imgPt, axis=1)**2)
        totalErr += err
    if mode == 'mean':
        if returnPts:
            return totalErr / float(imgPts.shape[0]), np.array(dis_projPts_all)
        else:
            return totalErr / float(imgPts.shape[0])
    else:
        if returnPts:
            return totalErr, np.array(dis_projPts_all)
        else:
            return totalErr

def distortPts(projPts, K, distortion):
    k1, k2 = distortion
    u0, v0 = K[0,2], K[1,2]
    fx, fy = K[0,0], K[1,1]
    projPts_normal = normalizeProjectPts(np.copy(projPts), u0, v0, fx, fy)
    dis_projPts = []
    for (projPt_normal, projPt) in zip(projPts_normal, projPts):
        u = projPt[0] + (projPt[0] - u0) * ( k1*(projPt_normal[0]**2+projPt_normal[1]**2) + k2*(projPt_normal[0]**2+projPt_normal[1]**2)**2 )
        v = projPt[1] + (projPt[1] - v0) * ( k1*(projPt_normal[0]**2+projPt_normal[1]**2) + k2*(projPt_normal[0]**2+projPt_normal[1]**2)**2 )
        dis_proj = np.array([u, v])
        dis_projPts.append(dis_proj)
    dis_projPts = np.array(dis_projPts)
    return dis_projPts

def normalizeProjectPts(projPts, u0, v0, fx, fy):
    center = np.array([[u0, v0]])
    projPts = projPts - center
    # projPts /= np.max(np.abs(projPts), axis=0)
    projPts /= np.array([fx, fy])
    return projPts


def undistortImage(imgPath, savePath, K, distort, pts, plot=True, show=False):
    for (pt, filePath) in zip(pts, imgPath):
        img = cv2.imread(filePath)
        if plot:
            for p in pt:
                cv2.circle(img, (int(p[0]), int(p[1])), 6, (0,0,255), -1)
        distCoeffs = np.array([distort[0], distort[1], 0, 0])
        img_undis = cv2.undistort(img, K, distCoeffs)
        if plot:
            for p in pt:
                cv2.circle(img_undis, (int(p[0]), int(p[1])), 6, (0,255,0), 2)
        cv2.imwrite(savePath+filePath[17:], img_undis)
        if show:
            cv2.imshow(filePath, img_undis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":

    files = sorted(glob.glob('Calibration_Imgs/*.jpg'))
    gridSize = (9, 6)

    mdPts = createModelPoints()

    _, imgPts = extractImagePoints(files)

    H = findHomography(mdPts, imgPts)

    K, P = estimateCameraParameters(H)

    err = reprojectError(mdPts, imgPts, K, P)
    print(err)