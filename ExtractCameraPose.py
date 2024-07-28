import numpy as np

def Camera_pose(E):
    
    U, _, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Translation vectors
    C_1 = U[:, 2]
    C_2 = -U[:, 2]
    C_3 = U[:, 2]
    C_4 = -U[:, 2]

    # Rotation matrices
    R_1 = U @ W @ V_T
    R_2 = U @ W @ V_T
    R_3 = U @ W.T @ V_T
    R_4 = U @ W.T @ V_T

    R_list = []
    C_list = []
    for R, C in zip([R_1, R_2, R_3, R_4], [C_1, C_2, C_3, C_4]):
        U_R, _, V_T_R = np.linalg.svd(R)
        det_R = np.linalg.det(U_R @ V_T_R)
        if det_R < 0:
            R = -R
            C = -C
        R_list.append(R)
        C_list.append(C)

    C_set = np.asarray(C_list)
    R_set = np.asarray(R_list)
    
    return C_set, R_set