import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

class PySBA:

    def __init__(self, proj_set, points3D, points2D, cameraIndices, point2DIndices, K):
        cameraArray = []
        for P in proj_set:
            R = P[:, :3]
            T = P[:, 3]
            R_quat, _ = cv2.Rodrigues(R)
            cameraArray.append(np.concatenate((R_quat.flatten(), T.flatten())))
        self.cameraArray = np.array(cameraArray)
        print("cameraPara: ", self.cameraArray.shape)
        self.points3D = points3D
        print("Point3d: ", points3D.shape)

        points2D_homo = np.concatenate((points2D, np.ones((len(points2D), 1))), axis=1)
        self.points2D = np.matmul(points2D_homo, np.linalg.inv(K).T)[..., :2]
        print("Point2d: ", self.points2D.shape)

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

    def rotate(self, points, rot_vecs):
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        diff = points_proj - points_2d
        print(np.mean(np.linalg.norm(diff, axis=1)))
        return diff.ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numCameras * 6 + numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)
        for s in range(6):
            A[2 * i, cameraIndices * 6 + s] = 1
            A[2 * i + 1, cameraIndices * 6 + s] = 1

        for s in range(3):
            A[2 * i, numCameras * 6 + pointIndices * 3 + s] = 1
            A[2 * i + 1, numCameras * 6 + pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        optimal_proj_set = []
        for param in camera_params:
            R, _ = cv2.Rodrigues(param[:3][:, None])
            T = param[3:]
            proj = np.concatenate((R, T[:,None]), axis=1)
            optimal_proj_set.append(proj)

        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return np.array(optimal_proj_set), points_3d


    def bundleAdjust(self):
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = np.hstack((self.cameraArray.ravel(), self.points3D.ravel()))
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

        params = self.optimizedParams(res.x, numCameras, numPoints)

        return params

class PySBA2:

    def __init__(self, proj_set, points3D, points2D, cameraIndices, point2DIndices, K):
        cameraArray = []
        for P in proj_set:
            R = P[:, :3]
            T = P[:, 3]
            R_quat, _ = cv2.Rodrigues(R)
            cameraArray.append(np.concatenate((R_quat.flatten(), T.flatten())))
        self.cameraArray = np.array(cameraArray)
        print("cameraPara: ", self.cameraArray.shape)
        self.points3D = points3D
        print("Point3d: ", points3D.shape)

        points2D_homo = np.concatenate((points2D, np.ones((len(points2D), 1))), axis=1)
        self.points2D = np.matmul(points2D_homo, np.linalg.inv(K).T)[..., :2]
        print("Point2d: ", self.points2D.shape)

        self.cameraIndices = cameraIndices
        self.point2DIndices = point2DIndices

    def rotate(self, points, rot_vecs):
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


    def project(self, points, cameraArray):
        points_proj = self.rotate(points, cameraArray[:, :3])
        points_proj += cameraArray[:, 3:6]
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
        return points_proj


    def fun(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        camera_params = self.cameraArray
        points_3d = params.reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices])
        diff = points_proj - points_2d
        print(np.mean(np.linalg.norm(diff, axis=1)))
        return diff.ravel()

    def bundle_adjustment_sparsity(self, numCameras, numPoints, cameraIndices, pointIndices):
        m = cameraIndices.size * 2
        n = numPoints * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cameraIndices.size)

        for s in range(3):
            A[2 * i, pointIndices * 3 + s] = 1
            A[2 * i + 1, pointIndices * 3 + s] = 1

        return A


    def optimizedParams(self, params, n_cameras, n_points):


        points_3d = params.reshape((n_points, 3))

        return points_3d


    def bundleAdjust(self):
        numCameras = self.cameraArray.shape[0]
        numPoints = self.points3D.shape[0]

        x0 = self.points3D.ravel()
        f0 = self.fun(x0, numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D)

        A = self.bundle_adjustment_sparsity(numCameras, numPoints, self.cameraIndices, self.point2DIndices)

        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(numCameras, numPoints, self.cameraIndices, self.point2DIndices, self.points2D))

        params = self.optimizedParams(res.x, numCameras, numPoints)

        return params

def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, cameraArray):
    points_proj = rotate(points, cameraArray[:, :3])
    points_proj += cameraArray[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    return points_proj


if __name__ == '__main__':
    pts3d = np.array([1,2,3,1])
    pts3d_2 = np.array([1,2,3])
    P = np.array([[ 0.97704353,  0.04764262, -0.20764422,  0.41525017],
     [-0.01007752,  0.9839182,   0.17833514,  0.0913554 ],
     [ 0.21280128, -0.17214866,  0.96181103, -0.85961669]])
    pts2d_1 = np.matmul(pts3d, P.T)

    R = P[:, :3]
    T = P[:, 3]
    R_quat, _ = cv2.Rodrigues(R)
    cameraArray = np.concatenate((R_quat.flatten(), T.flatten()))[None, ...]
    pts2d_2 = project(pts3d_2[None, ...], cameraArray)
    print(pts2d_1[:2]/pts2d_1[2])
    print(pts2d_2)
