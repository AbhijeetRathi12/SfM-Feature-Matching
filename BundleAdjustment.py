import time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix
from BuildVisibilityMatrix import VisibilityMatrix

def getRotVec(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def get2DPointsFromVisibilityMatrix(X_index, visibility_matrix, features_u, features_v):

    points2D = []
    feature_u_vis = features_u[X_index]
    feature_v_vis = features_v[X_index]

    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i,j] == 1:
                pt2D = np.hstack((feature_u_vis[i,j], feature_v_vis[i,j]))
                points2D.append(pt2D)

    return np.array(points2D).reshape(-1, 2) 

def getCameraPointIndexFromVisMatrix(visibility_matrix):
    
    camera_index = []
    point_index = []
    for i in range(visibility_matrix.shape[0]):
        for j in range(visibility_matrix.shape[1]):
            if visibility_matrix[i,j] == 1:
                camera_index.append(j)
                point_index.append(i)

    return np.array(camera_index).reshape(-1), np.array(point_index).reshape(-1)

def bundle_adjustment_sparsity(X_f, inlier_feature_flag, camIndex):

    X_index, visibility_matrix = VisibilityMatrix(X_f.reshape(-1), inlier_feature_flag, camIndex)


    m = np.sum(visibility_matrix) * 2
    n = (camIndex + 1) * 6 + len(X_index[0]) * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(np.sum(visibility_matrix))
    camera_index, point_index = getCameraPointIndexFromVisMatrix(visibility_matrix)

    for s in range(6):
        A[2 * i, camera_index * 6 + s] = 1
        A[2 * i + 1, camera_index * 6 + s] = 1

    for s in range(3):
        A[2 * i, (camIndex)* 6 + point_index * 3 + s] = 1
        A[2 * i + 1, (camIndex) * 6 + point_index * 3 + s] = 1

    return A

def rotate(points, rot_vecs):

    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def projectPoint(R, C, pt3D, K):
        P2 = np.dot(K, np.dot(R, np.hstack((np.identity(3), -C.reshape(3,1)))))
        x3D_h = np.hstack((pt3D, 1))
        x_proj = np.dot(P2, x3D_h.T)
        x_proj /= x_proj[-1]
        return x_proj

def project(points, camera_params, K):
    
    x_proj = []
    for i in range(len(camera_params)):
        R = getRotation(camera_params[i, :3], 'e')
        C = camera_params[i, 3:].reshape(3,1)
        pt3D = points[i]
        pt_proj = projectPoint(R, C, pt3D, K)[:2]
        x_proj.append(pt_proj)    
    return np.array(x_proj)

def fun(x0, camIndex, n_points, camera_indices, point_indices, points_2d, K):

    number_of_cam = camIndex + 1
    camera_params = x0[:number_of_cam * 6].reshape((number_of_cam, 6))
    points_3d = x0[number_of_cam * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices], K)
    error_vec = (points_proj - points_2d).ravel()
    
    return error_vec

def BundleAdjustment(X_complete, X_f, features_u, features_v, inlier_feature_flag, Rset, Cset, K, camIndex):
    
    X_index, visiblity_matrix = VisibilityMatrix(X_f, inlier_feature_flag, camIndex)
    print("Visibility Matrix: ")
    print(visiblity_matrix)
    
    pts_3d = X_complete[X_index]
    pts_2d = get2DPointsFromVisibilityMatrix(X_index, visiblity_matrix, features_u, features_v)

    RC_list = []
    for i in range(camIndex + 1):
        C, R = Cset[i], Rset[i]
        
        Q = Rotation.from_matrix(R).as_quat()
        if C.shape == (3, 1):
            RC = [Q[0], Q[1], Q[2], C[0][0], C[1][0], C[2][0]]
        else:
            RC = [Q[0], Q[1], Q[2], C[0], C[1], C[2]]
        RC_list.append(RC)

    
    RC_array = np.array(RC_list).reshape(-1, 6)
    
    x0 = np.hstack((RC_array.ravel(), pts_3d.ravel()))
    n_points = pts_3d.shape[0]

    camera_index, point_index = getCameraPointIndexFromVisMatrix(visiblity_matrix)
    
    A = bundle_adjustment_sparsity(X_f, inlier_feature_flag, camIndex)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-10, method='trf',
                        args=(camIndex, n_points, camera_index, point_index, pts_2d, K))
    t1 = time.time()

    print(f"Time taken : {t1 - t0}")
    print(f"Shape of A : {A.shape}")
    
    x1 = res.x
    num_camera = camIndex + 1
    camera_val_opt = x1[:num_camera * 6].reshape((num_camera, 6))
    points3D_opt = x1[num_camera * 6:].reshape((n_points, 3))

    X_complete_opt = np.zeros_like(X_complete)
    X_complete_opt[X_index] = points3D_opt

    Cset_opt, Rset_opt = [], []
    for i in range(len(camera_val_opt)):

        R = camera_val_opt[i,: 3]
        R = Rotation.from_rotvec(R).as_matrix()
        C = camera_val_opt[i, 3:].reshape(3,1)
        Cset_opt.append(C)
        Rset_opt.append(R)
    
    return Cset_opt, Rset_opt, X_complete_opt




    