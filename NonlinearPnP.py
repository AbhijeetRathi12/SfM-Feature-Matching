import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def error_function(X, K, feature_points, X_points):
    
    Q = X[:4]
    t = X[4:]

    R = Rotation.from_quat(Q).as_matrix()
    P = np.dot(K, np.hstack([R, t.reshape(3,1)]))
    
    errors = []
    X_points = np.array([np.append(row, 1) for row in X_points])
    for X, point in zip(X_points, feature_points):
        projected_point = P @ X
        u_projected = projected_point[0] / projected_point[2]
        v_projected = projected_point[1] / projected_point[2]
        u, v = point
        
        error = (u - u_projected) ** 2 + (v - v_projected) ** 2
        errors.append(error)
    
    error_total = np.mean(errors)
    return error_total

def Non_linear_PnP(K, feature_points, X_points, R, C):
    
    
    Q = Rotation.from_matrix(R).as_quat()
    X = np.concatenate([Q, C])

    optimized_params = least_squares(fun=error_function, x0=X, method="trf", args=[K, feature_points, X_points])
    
    Q_opt = optimized_params.x[:4]
    C_opt = optimized_params.x[4:]
    R_opt = Rotation.from_quat(Q_opt).as_matrix()
    
    return R_opt, C_opt