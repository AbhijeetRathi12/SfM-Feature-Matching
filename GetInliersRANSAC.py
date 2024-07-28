import numpy as np
from EstimateFundamentalMatrix import Fundamental_Matrix

def Inliers_RANSANC(points1, points2, index):

    iterations = 1000
    epsilon = 0.05
    percent_good_matches = 0.99
    maximum = 0
    
    num_matches = len(points1)
    latest_F = np.zeros((3, 3))

    best_matches = []
    inlier_index = []
    
    for _ in range(iterations):
        
        # Select 8 matched feature pairs from each image at random
        points = [np.random.randint(0, num_matches) for num in range(8)]
        
        pt_1 = points1[points[0]]
        pt_2 = points1[points[1]]
        pt_3 = points1[points[2]]
        pt_4 = points1[points[3]]
        pt_5 = points1[points[4]]
        pt_6 = points1[points[5]]
        pt_7 = points1[points[6]]
        pt_8 = points1[points[7]]
        pt_p_1 = points2[points[0]]
        pt_p_2 = points2[points[1]]
        pt_p_3 = points2[points[2]]
        pt_p_4 = points2[points[3]]
        pt_p_5 = points2[points[4]]
        pt_p_6 = points2[points[5]]
        pt_p_7 = points2[points[6]]
        pt_p_8 = points2[points[7]]

        pts = np.array([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8], np.float32)
        pts_prime = np.array([pt_p_1, pt_p_2, pt_p_3, pt_p_4, pt_p_5, pt_p_6, pt_p_7, pt_p_8], np.float32)
        
        F = Fundamental_Matrix(pts, pts_prime)

        num_good_matches = 0

        good_matches = []
        S_index = []
        # Compute inliers or best matches using |x2 * F * x1| < threshold, repeat until sufficient matches found
        for i in range(num_matches):
            
            x1j = np.array([points1[i, 0], points1[i, 1], 1])
            x2j = np.array([points2[i, 0], points2[i, 1], 1])

            F_pt = F @ x1j.T
            F_pt = F_pt.T

            pt_prime_F_pt = np.multiply(x2j, F_pt)

            error = np.sum(pt_prime_F_pt)

            if abs(error) < epsilon:
                num_good_matches += 1
                good_match = [[x1j[0], x1j[1]], [x2j[0], x2j[1]]]
                good_matches.append(good_match)
                S_index.append(index[i])

        if maximum < num_good_matches:

            maximum = num_good_matches

            good_matches = np.asarray(good_matches)
            matches_p = good_matches[:, 0]
            matches_p_prime = good_matches[:, 1]

            # Compute the fundamental matrix for the matched pairs
            latest_F = Fundamental_Matrix(matches_p, matches_p_prime)

            # Set for the output array of best matched points
            best_matches = good_matches
            inlier_index = S_index

            if num_good_matches > percent_good_matches * num_matches:       # end if desired matches num were found
                break

    return latest_F, best_matches, inlier_index