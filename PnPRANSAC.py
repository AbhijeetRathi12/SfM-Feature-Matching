import numpy as np
import random
from LinearPnP import Linear_PnP

def PnP_RANSAC(K, feature_points, X_points):

    iterations = 5000
    threshold = 10
    num_points = feature_points.shape[0]

    # Initialize variables to hold the best inliers found
    best_inliers = []
    X_points = np.array([np.append(row, 1) for row in X_points])
    
    for _ in range(iterations):
        # Select 6 points at random
        random_indices = random.sample(range(num_points), 6)
        x_hat = feature_points[random_indices, :]
        X_hat = X_points[random_indices, :]

        # Perform linear PnP to calculate the pose of the camera
        R, C = Linear_PnP(K, x_hat, X_hat)
        C = C.reshape((3, 1))
        
        # Project world points onto the image plane
        P = K @ np.concatenate((R, C), axis=1)
        
        projected_points = P @ X_points.T
        uvs = projected_points[:2] / projected_points[2:]

        # Calculate errors and identify inliers
        errors = np.sqrt(np.sum((uvs.T - feature_points)**2, axis=1))
        inliers = np.nonzero(errors < threshold)[0]

        # Update best inliers if necessary
        if len(inliers) > len(best_inliers):
            best_inliers = inliers.tolist()

    # Re-estimate R and C using all the best inliers
    feature_inliers = feature_points[best_inliers]
    world_point_inliers = X_points[best_inliers]
    R, C = Linear_PnP(K, feature_inliers, world_point_inliers)
    
    if np.linalg.det(R) < 0:
        R = -R
        C = -C

    return R, C