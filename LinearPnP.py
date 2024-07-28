import numpy as np

def Linear_PnP(K, x_hat, X_hat):

    u, v = x_hat[:, 0], x_hat[:, 1]
    X, Y, Z = X_hat[:, 0], X_hat[:, 1], X_hat[:, 2]

    A_list= []
    for i in range(len(X)):
        A1 = np.hstack([X[i], Y[i], Z[i], 1.0, 0.0, 0.0, 0.0, 0.0, -u[i] * X[i], -u[i] * Y[i], -u[i] * Z[i], -u[i]])
        A2 = np.hstack([0.0, 0.0, 0.0, 0.0, X[i], Y[i], Z[i], 1.0, -v[i] * X[i], -v[i] * Y[i], -v[i] * Z[i], -v[i]])
        A_point = np.vstack([A1, A2])
        A_list.append(A_point)
    
    A = np.vstack(A_list)
        
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1, :].reshape((3, 4))
    
    R_camera = P[0:3, 0:3]
    K_inv = np.linalg.inv(K)
    R = K_inv @ R_camera

    UR, DR, VtR = np.linalg.svd(R)                 
    R = UR @ VtR                                     

    C = P[:, 3]
    
    C = -R.T @ C
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C

    return R, C
