import numpy as np
from scipy.optimize import least_squares

def error_function(x, uv_1, uv_2, P, PO, point_index):
    point_1 = uv_1[point_index]
    point_2 = uv_2[point_index]
    
    # Calculate reprojected points
    point_reproj1 = PO @ x
    point_reproj2 = P @ x
    
    # Normalize reprojected points
    point_reproj1 /= point_reproj1[2]
    point_reproj2 /= point_reproj2[2]

    # Calculate squared errors
    error_reproj_1 = (point_reproj1[:2] - point_1) ** 2
    error_reproj_2 = (point_reproj2[:2] - point_2) ** 2

    # Concatenate error vectors
    error = np.concatenate((error_reproj_1, error_reproj_2))

    return error

def Non_linear_triangulation(K, C0, R0, C, R, x1, x2, X_points):
    
    num_points = len(X_points)
    
    uv_1 = x1
    uv_2 = x2

    X_points = np.hstack((X_points, np.ones((num_points, 1))))

    C = C.reshape((3, 1))
    C0 = C0.reshape((3, 1))        
    IC = np.append(R0, -C, axis=1)

    P = K @ R @ IC

    IC0 = np.append(R0, C0, axis=1)
    P0 = K @ R0 @ IC0

    X_pts = []

    # Optimize every point in X_points to get the list of refined points
    for i in range(num_points):
        x0 = X_points[i]

        # Perform non-linear least squares optimization
        result = least_squares(lambda x: error_function(x, uv_1, uv_2, P, P0, i), x0=x0, method='trf')

        # Divide by the last value to homogenize
        X_opt = result.x / result.x[-1]

        X_pts.append(X_opt)
        
    X_pts = np.asarray(X_pts)
    
    return X_pts