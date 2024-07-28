import numpy as np

def Disambiguate_camera(C_list, R_list, X_points_poses):
    num_poses = len(R_list)
    total_counts = np.zeros(num_poses, dtype=int) 
    
    for i in range(num_poses):
        R, C, X_points = R_list[i], C_list[i], np.array(X_points_poses[i])  
        r_3 = R[:, 2] 
        
        dots = np.dot(X_points - C.reshape(1, -1), r_3)
        
        # Check conditions in one go for all points
        cond_1 = dots > 0  # Check cheirality condition
        cond_2 = X_points[:, 2] > 0  # Check if Z coordinate is positive
        
        # Count points satisfying both conditions
        total_counts[i] = np.sum(cond_1 & cond_2)
    
    # Find index of pose with maximum count
    idx = np.argmax(total_counts)
    
    # Extract correct pose info
    C_correct, R_correct, X_correct = C_list[idx], R_list[idx], X_points_poses[idx]
    
    return C_correct, R_correct, X_correct