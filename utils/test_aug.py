#!/usr/bin/env python3

import cv2
import ast
import numpy as np
import open3d as o3d

from prediction.loader import get_data
from utils.transform import calc_tf_delta, pose_to_mat
from utils.realsense_utils import pcd_from_rgbd, Intrinsic640
from utils.data_aug import *

import argparse
import numpy as np
import matplotlib.pyplot as plt

################################################################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/cube_data_5_13_frame_1_2_7")
    parser.add_argument("--no_gripper", action="store_true")
    parser.add_argument("--crop_pic", action="store_true")
    parser.add_argument("--translate_pic", action="store_true")
    args = parser.parse_args()
    
    # data = get_data(["data/raw/cube_data_5_13_frame_1_2_7"], shuffle=False)
    # data = get_data(["data/raw/mouse_data_5_18_frame_1_2_6"], shuffle=False)
    # data = get_data(["data/raw/mouse_data_5_18_frame_1_2_3"], shuffle=False)

    intr = Intrinsic640()
    data = get_data([args.data], shuffle=False)
    print("Number of samples", len(data))
    print("Sample keys", data[0])

    index = 1

    if args.crop_pic:
        # load image
        rgb_img = cv2.imread(data[index][1]["rgb"], cv2.IMREAD_COLOR)
        depth_img = cv2.imread(data[index][1]["depth"], cv2.IMREAD_ANYDEPTH)
        
        # crop image
        cropped_img, _, mat = crop_imgs(rgb_img, depth_img, 220, 140, 650, 540, intr)
        
        # show concatenated image
        cv2.imshow("ori", rgb_img)
        cv2.imshow("cropped", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    if args.translate_pic:
        # load image
        rgb_img = cv2.imread(data[index][1]["rgb"], cv2.IMREAD_COLOR)
        depth_img = cv2.imread(data[index][1]["depth"], cv2.IMREAD_ANYDEPTH)
        
        trans_x = 100
        trans_y = -100
        rgb_img, depth_img, _ = translate_imgs(rgb_img, depth_img, trans_x, trans_y, intr)

        # Assuming depth_img is the depth image you want to visualize
        # Normalize the depth values between 0 and 1
        normalized_depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img))

        # Apply a colormap to the normalized depth image
        colormap = plt.get_cmap('viridis')  # You can choose a different colormap if desired
        depth_img_color = colormap(normalized_depth_img)

        # Display the depth image
        # cv2.imshow("rgb img", rgb_img)
        # Plot the RGB image in the first subplot
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(rgb_img)
        axs[0].axis('off')
        axs[0].set_title('RGB Image')
        # Plot the colorized depth image in the second subplot
        axs[1].imshow(depth_img_color)
        axs[1].axis('off')
        axs[1].set_title('Colorized Depth Image')
        plt.tight_layout()
        plt.show()
        exit()

    ref_joint = get_joints(data[index][1]) # initial

    i_delta, f_delta = get_transform(data[index], ref_joint)
    i_pcd, f_pcd = get_pcd(data[index], intr)
    i_pcd.transform(i_delta)
    # f_pcd.transform(f_delta)
    # Load point cloud
    # pcd = o3d.io.read_point_cloud("viz_pcd.ply")
    # Convert point cloud to voxel
    combined_pcd = i_pcd
    if args.no_gripper:
        combined_pcd = remove_gripper(combined_pcd)

    # Convert Open3D.o3d.geometry.PointCloud to numpy array
    print("number of points",  np.asarray(combined_pcd.points).shape)
    # cropped_pcd = combined_pcd.crop(bounding_box.inversed())
    # print("number of points",  np.asarray(cropped_pcd.points).shape)

    for i in [8, 15, 20, 4, 28, 4]:
        i_delta, f_delta = get_transform(data[i], ref_joint)
        i_pcd, f_pcd = get_pcd(data[i], intr)
        if args.no_gripper:
            i_pcd = remove_gripper(i_pcd)
        i_pcd.transform(i_delta)
        # f_pcd.transform(f_delta)
        # Load point cloud
        # pcd = o3d.io.read_point_cloud("viz_pcd.ply")
        # Convert point cloud to voxel
        combined_pcd += i_pcd

    # print number of points
    print("number of points",  np.asarray(combined_pcd.points).shape)
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.001)
    print("number of points",  np.asarray(combined_pcd.points).shape)
    # TODO: remove outliers
    cl , ind = combined_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.5)
    combined_pcd = combined_pcd.select_by_index(ind)
    print("number of points",  np.asarray(combined_pcd.points).shape)

    # visaualize combined point cloud
    o3d.visualization.draw_geometries([combined_pcd])

    # # Visualize voxel grid
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(combined_pcd, voxel_size=0.001)
    # o3d.visualization.draw_geometries([voxel_grid])
    # o3d.visualization.draw_geometries([pcd])
