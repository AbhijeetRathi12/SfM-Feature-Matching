import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def Read_Images(data_path):

    print(f'Reading images from "{data_path}"')
    images = []

    for file in os.listdir(data_path):
        
        if file.endswith('.png'):
            image = cv2.imread(f'{data_path}/{file}')
            images.append(image)
    
    images = np.array(images)
    return images

def Remove_duplicate_lines(data_path, images):
    
    for image_number in range(1, images.shape[0]):
        
        output_file = f'{data_path}/matchingnew{image_number}.txt'
        matching_file_path = f'{data_path}/matching{image_number}.txt'
        with open(matching_file_path, 'r') as f:
            lines = f.readlines()

        # Remove duplicates while preserving order
        unique_lines = []
        seen = set()
        for line in lines:
            stripped_line = line.strip()  # Remove leading and trailing whitespace
            if stripped_line not in seen:
                unique_lines.append(line)
                seen.add(stripped_line)

        with open(output_file, 'w') as f:
            f.writelines(unique_lines)

def Read_matching_file(data_path, images):
    
    features_1 = []
    features_2 = []
    rgb_feature = []
    for image_number in range(1, images.shape[0]):
        
        matching_file_path = f'{data_path}/matchingnew{image_number}.txt'
        with open(matching_file_path, 'r') as f:
            next(f)
            for line in f:
                line_items = np.array([float(num) for num in line.split()])
                n_matches = line_items[0]
                r, g, b = line_items[1:4]
                curr_u, curr_v = line_items[4:6]
                urow, vrow, flag = np.zeros(images.shape[0]), np.zeros(images.shape[0]), np.zeros(images.shape[0], dtype=int)
                urow[image_number - 1], vrow[image_number - 1], flag[image_number - 1] = curr_u, curr_v, 1
                match_idx = 1
                while n_matches > 1:
                    curr_img, curr_img_u, curr_img_v = int(line_items[5 + match_idx]), line_items[6 + match_idx], line_items[7 + match_idx]
                    match_idx += 3
                    urow[curr_img - 1], vrow[curr_img - 1], flag[curr_img - 1], n_matches = curr_img_u, curr_img_v, 1, n_matches - 1
                features_1.append(np.transpose(urow))
                features_2.append(np.transpose(vrow))
                rgb_feature.append(np.transpose(flag))
                
    return np.array(features_1).reshape(-1, images.shape[0]), np.array(features_2).reshape(-1, images.shape[0]), np.array(rgb_feature).reshape(-1, images.shape[0])

def Same_size_Images(images):
    
    imgs = images.copy()
    image_sizes = np.array([img.shape for img in imgs])
    max_shape = np.max(image_sizes, axis=0)
    
    resized_images = []

    for i, img in enumerate(imgs):
        resized_img = np.zeros(max_shape, np.uint8)
        resized_img[0: image_sizes[i, 0], 0: image_sizes[i, 1], 0: image_sizes[i, 2]] = img
        resized_images.append(resized_img)
    
    resized_images = np.array(resized_images)
    return resized_images

def Display_matches(image1, image2, pts1, pts2, file_path):
    
    img1, img2 = Same_size_Images([image1, image2])
    points1 = pts1.copy()
    points2 = pts2.copy()

    concat = np.concatenate((img1, img2), axis = 1)

    features_1u = points1[:, 0].astype(int)
    features_1v = points1[:, 1].astype(int)
    features_2u = points2[:, 0].astype(int) + img1.shape[1]
    features_2v = points2[:, 1].astype(int)

    for i in range(len(features_1u)):
        cv2.line(concat, (features_1u[i], features_1v[i]), (features_2u[i], features_2v[i]), color=(255,0,0), thickness=1)
        cv2.circle(concat, (features_1u[i], features_1v[i]), 2, color=(0,0,255), thickness=1)
        cv2.circle(concat, (features_2u[i], features_2v[i]), 2, color=(0,255,255), thickness=1)
    cv2.imwrite(file_path, concat)

def Plot_Epipoles(image1, image2, points1, points2, F, filename1, filename2):
    
    e1 = np.linalg.svd(F)[2][-1]
    e2 = np.linalg.svd(F.T)[2][-1]

    e1 = e1 / e1[2]
    e2 = e2 / e2[2]

    img1 = image1.copy()
    for i in range(len(points1)):
        cv2.line(img1, points1[i].astype(int), tuple(e1[:2].astype(int)), (255, 0, 0), 1)

    cv2.imwrite(filename1, img1)

    img2 = image2.copy()
    for i in range(len(points2)):
        cv2.line(img2, points2[i].astype(int), tuple(e2[:2].astype(int)), (255, 0, 0), 1)

    cv2.imwrite(filename2, img2)

def Reprojection_Error_PnP(X, x, K, R, C):

    P = np.dot(K, np.hstack((R, -np.dot(R, C.reshape(3, 1)))))

    E = []
    for X, pt in zip(X, x):
        
        p_1T, p_2T, p_3T = P# rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = np.hstack((X.reshape(1,-1), np.ones((X.reshape(1,-1).shape[0], 1)))).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        e = np.square(v - v_proj) + np.square(u - u_proj)

        E.append(e)
    
    mean_err = np.mean(np.array(E).squeeze())
    return mean_err

def Reprojection_Error(X, p1, p2, R1, C1, R2, C2, K):
    
    P1 = np.dot(K, np.hstack((R1, -np.dot(R1, C1.reshape(3, 1)))))
    P2 = np.dot(K, np.hstack((R2, -np.dot(R2, C2.reshape(3, 1)))))

    p1_1t, p1_2t, p1_3t = P1
    p1_1t, p1_2t, p1_3t = p1_1t.reshape(1,4), p1_2t.reshape(1,4), p1_3t.reshape(1,4)
    p2_1t, p2_2t, p2_3t = P2
    p2_1t, p2_2t, p2_3t = p2_1t.reshape(1,4), p2_2t.reshape(1,4), p2_3t.reshape(1,4)

    X = X.reshape(4,1)
    prj_pts1 = []
    prj_pts2 = []
    # Reprojection error w.r.t camera 1 ref
    u1, v1 = p1[0], p1[1]

    u1_proj = np.divide(p1_1t.dot(X), p1_3t.dot(X))
    v1_proj = np.divide(p1_2t.dot(X), p1_3t.dot(X))
    
    error1 =  np.square(v1 - v1_proj[0][0]) + np.square(u1 - u1_proj[0][0])
    
    prj_pts1.append([u1_proj[0][0], v1_proj[0][0]])
    # Reprojection error w.r.t camera 2 ref
    u2, v2 = p2[0], p2[1]

    u2_proj = np.divide(p2_1t.dot(X), p2_3t.dot(X))
    v2_proj = np.divide(p2_2t.dot(X), p2_3t.dot(X))

    error2 = np.square(v2 - v2_proj[0][0]) + np.square(u2 - u2_proj[0][0])
    prj_pts2.append([u2_proj[0][0], v2_proj[0][0]])
    
    return error1, error2, prj_pts1, prj_pts2

def Visualize_matches_RANSAC(num_image_1, num_image_2, matched_points, data_path, results_path):
    
    # Read images
    image1 = cv2.imread(f"{data_path}/{num_image_1}.png")
    image2 = cv2.imread(f"{data_path}/{num_image_2}.png")
    
    # Determine dimensions of the combined image
    max_height = max(image1.shape[0], image2.shape[0])
    total_width = image1.shape[1] + image2.shape[1]
    
    # Create blank combined image
    if len(image1.shape) == 3:
        combined_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    else:
        combined_image = np.zeros((max_height, total_width), dtype=np.uint8)
    
    # Copy images onto the combined image
    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    combined_image[:image2.shape[0], image1.shape[1]:] = image2
    
    # Mark matched points
    circle_size = 4
    for point1, point2 in matched_points:
        cv2.line(combined_image, (int(point1[0]), int(point1[1])), (int(point2[0] + image1.shape[1]), int(point2[1])), (0, 0, 255), 1)
        cv2.circle(combined_image, (int(point1[0]), int(point1[1])), circle_size, (255, 255, 0), 1)
        cv2.circle(combined_image, (int(point2[0]) + image1.shape[1], int(point2[1])), circle_size, (0, 255, 255), 1)
    
    # Resize for better displaying
    scale = 1.5
    resized_image = cv2.resize(combined_image, (0, 0), fx=1/scale, fy=1/scale)
    
    cv2.imwrite(f'{results_path}/After_RANSAC_{num_image_1}_{num_image_2}.png', resized_image)
    # cv2.imshow(f"After RANSAC -{num_image_1}-{num_image_2}", resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def Visualize_point_Linear(points_lists, results_path):
    
    colors = ["red", "blue", "green", "purple"]
    for points_array, color in zip(points_lists, colors):
        x_pts, _, z_pts = points_array.T  # Extract x and z coordinates
        plt.scatter(x_pts, z_pts, color=color, s=1)

    plt.title("Initial Triangulation")
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.savefig(f"{results_path}/Initial Triangulation.png")
    plt.show()
    
def Visualize_points_lin_nonlin(a, b, points_list_1, points_list_2, results_path):

    points_list_1 = np.asarray(points_list_1)
    points_list_2 = np.asarray(points_list_2)

    x_pts_1, _, z_pts_1 = points_list_1[:, 0], points_list_1[:, 1], points_list_1[:, 2]
    x_pts_2, _, z_pts_2 = points_list_2[:, 0], points_list_2[:, 1], points_list_2[:, 2]

    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="red", s=dot_size)
    plt.scatter(x_pts_2, z_pts_2, color="blue", s=dot_size)

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend(["Linear", "Nonlinear"])
    plt.savefig(f"{results_path}/Triangulated World Points{a}_{b}.png")
    plt.show()