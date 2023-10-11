import cv2
import ast
import numpy as np
import open3d as o3d

from stretch_remote.robot_utils import read_robot_status
from utils.transform import calc_tf_delta, pose_to_mat
from utils.realsense_utils import pcd_from_rgbd, Intrinsic640

import argparse
import numpy as np
import matplotlib.pyplot as plt

################################################################################

def load_robot_state(data_point):
    with open(data_point, 'r') as f:
        robot_state = read_robot_status(ast.literal_eval(f.read()))
    return robot_state

def get_joints(data_point_state):
    s = load_robot_state(data_point_state["state"])
    return [s["x"], s["y"], s["z"], s["roll"], s["pitch"], s["yaw"]]

def get_transform(data_point, ref_joint):
    i_joints = get_joints(data_point[1])
    f_joints = get_joints(data_point[2])
    i_delta_mat = calc_tf_delta(ref_joint, i_joints)
    f_delta_mat = calc_tf_delta(ref_joint, f_joints)
    return i_delta_mat, f_delta_mat

def crop_imgs(image, image2, x1, y1, x2, y2, intr):
    """
    Return cropped image and the transformation matrix
    :return: cropped image, transformation matrix
    """
    # get size of image
    h, w, _ = image.shape
    # TODO: use shrink_ratio, x_movement_ratio, y_move_ratio, intr):
    # new_h = int(h * shrink_ratio)
    # new_w = int(w * shrink_ratio)
    
    # # potential movement
    # delta_h = h - new_h
    # delta_w = w - new_w

    rx_1 = np.tan((x1-w/2)/intr.fx)
    rx_2 = np.tan((x2-w/2)/intr.fx)
    ry_1 = np.tan((y1-h/2)/intr.fy)
    ry_2 = np.tan((y2-h/2)/intr.fy)
    # print("rx_1: {}, rx_2: {}, ry_1: {}, ry_2: {}".format(rx_1, rx_2, ry_1, ry_2))
    
    # TODO: make it more better, convert the direction
    # TODO: include translation from ppx ppy
    avg_pitch = (rx_1 + rx_2) / 2
    avg_yaw = (ry_1 + ry_2) / 2
    mat = pose_to_mat([0, 0, 0, avg_pitch, avg_yaw, 0])
    # print("avg_pitch: {}, avg_yaw: {}".format(avg_pitch, avg_yaw))
    return image[y1:y2, x1:x2], image2[y1:y2, x1:x2], mat

def add_noise_to_borders(image, x, y, is_rgb=False):
    # Add noise only to the borders of the image
    if is_rgb:
        noise_img = np.random.random(image.shape).astype(np.float32)
        noise_img = (noise_img * 255).astype(np.uint8)
    else:
        # this is a depth image, get the max depth value and apply a small noise to the noise img
        max_depth = float(np.max(image))
        noise_img = np.random.random(image.shape).astype(np.float32)
        noise_img = (-noise_img*20000 + max_depth).astype(np.uint16) # the current depth image is uint16
   
    image = image.copy()
    # apply change the border pixels to noise according to the x, y translation
    if y > 0:
        image[:y, :] = noise_img[:y, :]
    elif y < 0:
        image[y:, :] = noise_img[y:, :]
    if x > 0:
        image[:, :x] = noise_img[:, :x]
    elif x < 0:
        image[:, x:] = noise_img[:, x:]
    return image

def translate_imgs(rgb_img, depth_img, x, y, intr,
                   rgb_noise=True, depth_noise=True):
    """
    :arg x: x translation in pixel +ve is right -ve is left
    :arg y: y translation in pixel +ve is down -ve is up
    :return: translated rgb and depth images, transformation matrix
    """
    rows, cols = rgb_img.shape[:2]
    affine_mat = np.float32([[1, 0, x],
                             [0, 1, y]])
    # Apply the translation matrix to shift the image right
    translated_rgb_img = cv2.warpAffine(
        rgb_img, affine_mat, (cols, rows), borderValue=(255, 255, 255))
    translated_depth_img = cv2.warpAffine(
        depth_img, affine_mat, (cols, rows), borderValue=float(np.max(depth_img)))
    
    if rgb_noise:
        translated_rgb_img = add_noise_to_borders(
            translated_rgb_img, x, y, is_rgb=True)
    if depth_noise:
        translated_depth_img = add_noise_to_borders(
            translated_depth_img, x, y, is_rgb=False)

    h, w, _ = rgb_img.shape
    ry = -np.tan(x/intr.fx) # x trans is changing the y-axis rotation of the cam
    rx = np.tan(y/intr.fy)  # y trans is changing the x-axis rotation of the cam
    # print("rx_1: {}, rx_2: {}, ry_1: {}, ry_2: {}".format(rx_1, rx_2, ry_1, ry_2))
  
    # TODO: include translation from ppx ppy? or undistort the image
    mat = pose_to_mat([0, 0, 0, rx, ry, 0])
    # print("cam roll: {}, pitch: {}".format(rx, ry))
    return translated_rgb_img, translated_depth_img, mat

################################################################################

def get_pcd(data_point, intr):
    i_pcd = pcd_from_rgbd(
        data_point[1]["rgb"], data_point[1]["depth"], intr)
    f_pcd = pcd_from_rgbd(
        data_point[2]["rgb"], data_point[2]["depth"], intr)
    return i_pcd, f_pcd

def remove_gripper(pcd):
    """remove a retangular region of the point cloud
    this is to remove the robot gripper"""
    invert = True # Invert the mask will then show the points to be removed
    pcd_np = np.asarray(pcd.points)
    xmin, xmax = -0.1, 0.1
    ymin, ymax = -0.1, 0.1
    zmin, zmax = 0., 0.22
    # Apply the condition to the points
    mask = np.logical_or(
        np.logical_or(
            np.logical_or(pcd_np[:, 0] < xmin, pcd_np[:, 0] > xmax),
            np.logical_or(pcd_np[:, 1] < ymin, pcd_np[:, 1] > ymax),
        ),
        np.logical_or(pcd_np[:, 2] < zmin, pcd_np[:, 2] > zmax),
    )
    # Apply the mask to remove points
    return pcd.select_by_index(np.where(mask == invert)[0])    
