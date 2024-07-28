import numpy as np

def Essential_matrix(F, K):
    
    E = K.T @ F @ K

    U,S,Vt = np.linalg.svd(E)

    S_new = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0 ,0]])
    
    E = U @ S_new @ Vt

    return E