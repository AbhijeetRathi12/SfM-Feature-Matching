import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from PnPRANSAC import PnP_RANSAC
from BundleAdjustment import BundleAdjustment
from NonlinearPnP import Non_linear_PnP
from GetInliersRANSAC import Inliers_RANSANC
from ExtractCameraPose import Camera_pose
from EstimateFundamentalMatrix import Fundamental_Matrix
from LinearTriangulation import Linear_triangulation
from NonlinearTriangulation import Non_linear_triangulation
from DisambiguateCameraPose import Disambiguate_camera
from EssentialMatrixFromFundamentalMatrix import Essential_matrix
from Utils import *


def Camera_matrix(data_path):
    
    calibration_file = (f"{data_path}/calibration.txt")
    
    with open(calibration_file, 'r') as f:
        lines = f.readlines()
    K = np.array([[float(x) for x in line.split()] for line in lines])
    
    return K

def visualize_points_camera_poses(X_before, X_bundle, R_set, C_set, results_path, name):

    X_before = np.asarray(X_before)
    X_bundle = np.asarray(X_bundle)
    x_pts_1, _, z_pts_1 = X_before[:, 0], X_before[:, 1], X_before[:, 2]
    x_pts_2, _, z_pts_2 = X_bundle[:, 0], X_bundle[:, 1], X_bundle[:, 2]
    dot_size = 1
    axes_lim = 20

    plt.scatter(x_pts_1, z_pts_1, color="blue", s=dot_size, label = 'Before Sparse Bundle Adj')
    plt.scatter(x_pts_2, z_pts_2, color="red", s=dot_size, label = 'After Sparse Bundle Adj')
    
    for i in range(len(R_set)):

        r2 = Rotation.from_matrix(R_set[i])
        angles2 = r2.as_euler("zyx", degrees=True)

        plt.plot(C_set[i][0], C_set[i][2], marker=(3, 0, int(angles2[1])), markersize=15, linestyle='None')

    plt.title("triangulated world points")
    plt.xlim(-axes_lim, axes_lim)
    plt.ylim(-5, 30)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend(loc = 'upper left')
    plt.savefig(f"{results_path}/ Before_After_Bundle_{name}.png")
    plt.show()

def visualize_reprojection(image_nums, err1_lists, err2_lists, name, data_path, results_path):
    
    for i in range(len(err1_lists)):
        err1_list = err1_lists
        err2_list = err2_lists
        image = cv2.imread(f"{data_path}/{image_nums}.png")
        for j in range(len(err1_list)):
            cv2.circle(image, (int(err1_list[j][0]), int(err1_list[j][1])), radius=2, color = (0, 0, 255), thickness = -1)
        for k in range(len(err2_list)):
            cv2.circle(image, (int(err2_list[k][0][0]), int(err2_list[k][0][1])), radius=2, color = (0, 255, 0), thickness = -1)
    
    cv2.imshow(f"{name}_reprojection visualizing -{image_nums}", image)
    cv2.waitKey(0)
    cv2.imwrite(f'{results_path}/{name}_reprojection_{image_nums}.png', image)
    cv2.destroyAllWindows()


def main():
    
    data_path = "Phase1/Data"
    results_path = (f"{data_path}/IntermediateOutputImages")

    # check if results folder already exists
    if not os.path.exists(results_path):
        print('Results path not found! Creating folder...')
        os.makedirs(results_path)
    
    images = Read_Images(data_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    Remove_duplicate_lines(data_path, images)
    
    features_u, features_v, flag_feature = Read_matching_file(data_path, images)
    
    print("Initializing RANSAC")
    
    F_array = np.empty(shape=(images.shape[0], images.shape[0]), dtype=object)
    inlier_feature_flag = np.zeros_like(flag_feature)
    F_list = []
    
    for i in range(images.shape[0]- 1):
        
        for j in range(i + 1, images.shape[0]):
            index = np.where(flag_feature[:, i] & flag_feature[:, j])
            pts1 = np.hstack((features_u[index, i].reshape((-1, 1)), features_v[index, i].reshape((-1, 1))))
            pts2 = np.hstack((features_u[index, j].reshape((-1, 1)), features_v[index, j].reshape((-1, 1))))
            Display_matches(images[i], images[j], pts1, pts2, file_path=f'{results_path}/Before RANSAC{i+1}{j+1}.png')
            index = np.reshape(np.array(index), -1)
            if len(index) > 8:
                F, best_matches, inlier_index = Inliers_RANSANC(pts1, pts2, index)
                print(f'Images {i+1} and {j+1}: {len(inlier_index)} inliers out of {len(index)} features.')
                Visualize_matches_RANSAC(i+1 , j+1, best_matches, data_path, results_path)
                F_list.append(F)
                inlier_feature_flag[inlier_index, i] = 1
                inlier_feature_flag[inlier_index, j] = 1
                
    print("RANSAC Complete")
    
    print("Calculating Fundamental Matrix")
    for i in range(images.shape[0]- 1):
            
        for j in range(i + 1, images.shape[0]):

            inlier_index = np.where(inlier_feature_flag[:,i] & inlier_feature_flag[:,j])
            inlier_pts1 = np.hstack((features_u[inlier_index, i].reshape((-1, 1)), features_v[inlier_index, i].reshape((-1, 1))))
            inlier_pts2 = np.hstack((features_u[inlier_index, j].reshape((-1, 1)), features_v[inlier_index, j].reshape((-1, 1))))

            F = Fundamental_Matrix(inlier_pts1, inlier_pts2)
            F_array[i][j] = F

            Plot_Epipoles(images[i], images[j], inlier_pts1, inlier_pts2, F, filename1=f'{results_path}/epipoles_F{i+1}{j+1}-{i+1}.png', filename2=f'{results_path}/epipoles_F{i+1}{j+1}-{j+1}.png')
    
    print("Fundamental Matrix Calculation Done")

    print("Registering Image 1 and Image 2")
    i, j = 0, 1
    F = F_list[0]
    print("The fundamental matrix for Image 1 and Image 2:")
    print(F)
    
    K = Camera_matrix(data_path)
    
    print("Calculating Essential Matrix")
    E = Essential_matrix(F, K)
    print("The essential matrix for Image 1 and Image 2:")
    print(E)
    
    print('Finding Camera Pose')
    Cset, Rset = Camera_pose(E)
    
    index = np.where(inlier_feature_flag[:,i] & inlier_feature_flag[:,j])
    
    x1 = np.hstack((features_u[index, i].reshape((-1, 1)), features_v[index, i].reshape((-1, 1))))
    x2 = np.hstack((features_u[index, j].reshape((-1, 1)), features_v[index, j].reshape((-1, 1))))

    print("Doing Linear Triangulation")
    
    pts3D_h = []
    C0 = np.zeros(3)
    R0 = np.identity(3)
    for i in range(len(Cset)):
        X = Linear_triangulation(K, C0, R0, Cset[i], Rset[i], x1, x2)
        pts3D_h.append(X)
    
    print("Linear Triangulation Complete")
    
    Visualize_point_Linear(pts3D_h, results_path)
    
    print("Disambiguating camera poses")
    C, R, X0 = Disambiguate_camera(Cset, Rset, pts3D_h)
    
    print("Doing Non-Linear Triangulation")
    X_optimized = Non_linear_triangulation(K, C0, R0, C, R, x1, x2, X0)
    print("Non-Linear Triangulation complete")
    
    X_ = np.array([np.append(row, 1) for row in X0])
    total_err1 = []
    projected_points1 = []
    projected_points2 = []
    projected_points1_linear = []
    projected_points2_linear = []
    for pt1, pt2, X_3d in zip(x1,x2,X_):
        
        err1, err2, prj_pts1, prj_pts2 = Reprojection_Error(X_3d,pt1,pt2,R0,C0,R,C,K)
        total_err1.append(err1+err2)
        projected_points1_linear.append(prj_pts1)
        projected_points2_linear.append(prj_pts2)
        
    mean_err1 = np.mean(total_err1)
    
    visualize_reprojection(1, x1, projected_points1_linear, "Linear", data_path, results_path)
    visualize_reprojection(2, x2, projected_points2_linear, "Linear", data_path, results_path)
    
    total_err2 = []
    for pt1, pt2, X_3d in zip(x1,x2,X_optimized):
        err1, err2, prj_pts1, prj_pts2 = Reprojection_Error(X_3d,pt1,pt2,R0,C0,R,C,K)
        total_err2.append(err1+err2)
        projected_points1.append(prj_pts1)
        projected_points2.append(prj_pts2)
    
    mean_err2 = np.mean(total_err2)
    
    visualize_reprojection(1, x1, projected_points1, "Non_linear", data_path, results_path)
    visualize_reprojection(2, x2, projected_points2, "Non_linear", data_path, results_path)
    print("Between images 1 and 2: ")
    print("Before optimization Non-Linear Triangulation: ", mean_err1, "After optimization Non-Linear Triangulation: ", mean_err2)
    
    Visualize_points_lin_nonlin(1, 2, X0, X_optimized, results_path)
    
    X_complete = np.zeros((len(features_u), 3))
    camera_index = np.zeros((len(features_u), 1), dtype=int)
    X_f = np.zeros((len(features_u), 1), dtype=int)
    
    XYZ = np.delete(X_optimized, X_optimized.shape[1]-1, axis=1)
    X_complete[index] = XYZ
    camera_index[index] = 1
    X_f[index] = 1
    
    # setting points below origin along z-axis as zero
    X_f[np.where(X_complete[:, 2] < 0)] = 0

    R_list = []
    C_list = []

    R_list.append(R0)
    R_list.append(R)
    C_list.append(C0)
    C_list.append(C)
    
    print("Cameras 1 and 2 registered")
    
    X_points_bundle = []
    X_points_before_bundle = []
    
    print("Registering remaining cameras")
    for i in range(2, images.shape[0]):

        print(f"Registering image {i+1}")
        
        feature_index_i = np.where(X_f[:, 0] & inlier_feature_flag[:, i])
        
        if len(feature_index_i[0]) < 8:
            print(f"Number of common points between X and image {i+1} : {len(feature_index_i)}")
            continue
        
        x = np.hstack((features_u[feature_index_i, i].reshape(-1, 1), features_v[feature_index_i, i].reshape(-1, 1)))
        X = X_complete[feature_index_i, :].reshape(-1, 3)
        
        print("Doing PnP RANSAC")
        Rnew, Cnew = PnP_RANSAC(K, x, X)
        print("PnP RANSAC complete")
        
        linearPnPError = Reprojection_Error_PnP(X, x, K, Rnew, Cnew)
        
        print("Doing Non-Linear PnP")
        Rnew, Cnew = Non_linear_PnP(K, x, X, Rnew, Cnew)
        print("Non-Linear PnP complete")
        
        nonLinearPnPError = Reprojection_Error_PnP(X, x, K, Rnew, Cnew)
        print(f"Error after linear PnP : {linearPnPError}")
        print(f"Error after non-linear PnP : {nonLinearPnPError}\n")
        
        C_list.append(Cnew)
        R_list.append(Rnew)

        for j in range(i):
            
            X_index = np.where(inlier_feature_flag[:, i] & inlier_feature_flag[:, j])
            if (len(X_index[0])) < 8:
                continue
            
            x1 = np.hstack((features_u[X_index, j].reshape((-1, 1)), features_v[X_index, j].reshape((-1, 1))))
            x2 = np.hstack((features_u[X_index, i].reshape((-1, 1)), features_v[X_index, i].reshape((-1, 1))))

            print("Doing Linear Triangulation")
            Xnew = Linear_triangulation(K, C_list[j], R_list[j], Cnew, Rnew, x1, x2)
            print("Linear Triangulation complete")
            Y = Xnew
            X_ = np.array([np.append(row, 1) for row in Xnew])
            total_err1 = []
            projected_points1i = []
            projected_points2i = []
            projected_points1i_linear = []
            projected_points2i_linear = []
            for pt1, pt2, X_3d in zip(x1,x2,X_):
                err1, err2, prj_pts1, prj_pts2 = Reprojection_Error(X_3d,pt1,pt2,R0,C0,Rnew,Cnew,K)
                total_err1.append(err1+err2)
                projected_points1i_linear.append(prj_pts1)
                projected_points2i_linear.append(prj_pts2)
            mean_err1 = np.mean(total_err1)
            # visualize_reprojection(i+1, j+1, x1, projected_points1i_linear, "linear", data_path, results_path)
            # visualize_reprojection(j+1, i+1, x2, projected_points2i_linear, "linear", data_path, results_path)

            print("Doing Non-Linear Triangulation")
            Xnew = Non_linear_triangulation(K, C_list[j], R_list[j], Cnew, Rnew, x1, x2, Xnew)
            print("Non-Linear Triangulation complete")
            Visualize_points_lin_nonlin(i+1, j+1, Y, Xnew, results_path)
            total_err2 = []
            for pt1, pt2, X_3d in zip(x1,x2,Xnew):
                err1, err2, prj_pts1, prj_pts2 = Reprojection_Error(X_3d,pt1,pt2,R0,C0,Rnew,Cnew,K)
                total_err2.append(err1+err2)
                projected_points1i.append(prj_pts1)
                projected_points2i.append(prj_pts2)
            mean_err2 = np.mean(total_err2)
            # visualize_reprojection(i+1, j+1, x1, projected_points1i, "Non_linear", data_path, results_path)
            # visualize_reprojection(j+1, i+1, x2, projected_points2i, "Non_linear", data_path, results_path)

            print(f"Between images{j+1} and {i+1}:")
            print(f"Before optimization Non-Linear Triangulation: ", mean_err1, "After optimization Non-Linear Triangulation: ", mean_err2)
    
            Xnew = np.delete(Xnew, Xnew.shape[1]-1, axis=1)
            X_complete[X_index] = Xnew
            X_f[X_index] = 1
            print(f"No of points between {j+1} and {i+1}: {len(X_index[0])}")
        
        plot_index = np.where(X_f[:, 0])
        X1 = X_complete[plot_index]
        X_points_before_bundle.append(X1)
        
        
        print("Doing Bundle Adjustment")
        C_list, R_list, X_complete = BundleAdjustment(X_complete, X_f, features_u, features_v, inlier_feature_flag, R_list, C_list, K, i)
        X_points_bundle.append(X_complete)
        for k in range(0,i+1):
                X_index = np.where(X_f[:,0] & inlier_feature_flag[:,k])
                x = np.hstack((features_u[X_index,k].reshape(-1,1), features_v[X_index,k].reshape(-1,1)))
                X = X_complete[X_index]
                BundAdj_error = Reprojection_Error_PnP(X,x,K,R_list[k],C_list[k])
                print("Error after Bundle Adjustment: ", BundAdj_error)
    
    feature_idx = np.where(X_f[:, 0])
    X_bundle = X_points_bundle[0][feature_idx]
    X_complt_bundle = X_complete[feature_idx]
    X_complt_before_bundle = X_points_before_bundle[-1]
    
    visualize_points_camera_poses(X_optimized, X_bundle, R_list, C_list, results_path, "1_3")
    visualize_points_camera_poses(X_complt_before_bundle, X_complt_bundle, R_list, C_list, results_path, "All")


if __name__ == '__main__':
    main()
    

