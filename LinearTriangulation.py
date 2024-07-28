import numpy as np

def Skew_matrix(x):
    
    X = np.array([[0, -x[2], x[1]],
                  [x[2], 0, -x[0]],
                  [-x[1], x[0], 0]])
    
    return X

def Linear_triangulation(K, C0, R0, C, R, x1, x2):
    
    uv_1 = x1
    uv_2 = x2

    ones = np.ones((uv_1.shape[0], 1))

    points_1 = np.concatenate((uv_1, ones), axis=1)
    points_2 = np.concatenate((uv_2, ones), axis=1)

    C = C.reshape((3, 1))
    C0 = C0.reshape((3, 1))                    
    IC = np.append(R0, -C, axis=1)

    P = K @ R @ IC

    IC0 = np.append(R0, C0, axis=1)
    P0 = K @ R0 @ IC0

    X_pts = []
    num_points = x1.shape[0]

    for i in range(num_points):

        X1_i = Skew_matrix(points_1[i]) @ P0
        X2_i = Skew_matrix(points_2[i]) @ P

        x_P = np.vstack((X1_i, X2_i))
        _, _, vt = np.linalg.svd(x_P)
        X_pt = vt[-1][:]

        X_pt = X_pt / X_pt[3]
        X_pt = X_pt[0:3]
        X_pts.append(X_pt)
    X_pts = np.asarray(X_pts)

    return X_pts