import numpy as np

def Fundamental_Matrix(vec1, vec2):
    
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    num_points = vec1.shape[0]
    
    # Construct matrix A
    x1, y1 = vec1[:, 0], vec1[:, 1]
    x2, y2 = vec2[:, 0], vec2[:, 1]
    A = np.vstack((x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, np.ones(num_points))).T

    # Perform SVD
    _, _, Vt = np.linalg.svd(A)
    
    # Extract fundamental matrix from the last row of Vt
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt

    return F