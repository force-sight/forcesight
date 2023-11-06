#!/usr/bin/env python3

from utils.ft_utils import ft_to_cam_rotation
from utils.realsense_utils import Intrinsic640, get_point_cloud
from utils.data_pipeline import *
from utils.pred_utils import *

import os
import cv2
import torch
import numpy as np
from robot.robot_utils import *
from utils.transform import *
from recording.ft import EEF_PITCH_WEIGHT_OFFSET

import matplotlib.pyplot as plt

def visualize_points(img, points, colors=[(0, 100, 255), (0, 255, 100)],
                     show_depth=True, show_point=True):
    # points is a list of numpy arrays of shape (3,) in the camera frame
    # we want to draw a circle at each point
    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()
    h, w, _ = img.shape

    # for point in points:
    for i in range(len(points)):
        color = colors[i]
        point = points[i]
        if point is None:
            continue
        point = point.reshape(3, 1) # (3,) -> (3, 1)
        # if points are Nan, skip
        if np.isnan(point).any():
            return img

        point2d = cv2.projectPoints(
            point, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
        
        if show_point:
            img = cv2.circle(img, (int(point2d[0]), int(point2d[1])), 3, color, -1)
        
        # clip the point that is out of the image
        point2d[0] = np.clip(point2d[0], 0, w)
        point2d[1] = np.clip(point2d[1], 0, h)
        # cv2.circle(img, (int(point2d[0]), int(point2d[1])), 3, color, thickness=-1)
    
        if show_depth:
            if i == 0:
                x_offset = int(point2d[0]) - 50 - 80
            else:
                x_offset = int(point2d[0]) - 50 + 80
            x_offset = np.clip(x_offset, 0, w)

            # add z-distance text to the image
            z_dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)[0]
            # print('z_dist: ', z_dist)
            cv2.putText(img, 
                        "D: {:.2f}m".format(z_dist),  # text
                        (x_offset, int(point2d[1] - 10)),  # bottom left coordinate
                        cv2.FONT_HERSHEY_SIMPLEX,  # font family
                        0.7,  # font size
                        (255, 200, 0),  # font color
                        2)  # font stroke
        # img = cv2.drawMarker(img, (int(point2d[0]), int(point2d[1])),
        #                      color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=3)
    
        # add z-distance text to the image
        # z_dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)[0]
        # # print('z_dist: ', z_dist)
        # cv2.putText(img, 
        #             "D: {:.2f}m".format(z_dist),  # text
        #             (int(point2d[0]), int(point2d[1] - 10)),  # bottom left coordinate
        #             cv2.FONT_HERSHEY_SIMPLEX,  # font family
        #             0.6,  # font size
        #             (255, 0, 0),  # font color
        #             2)  # font stroke
    return img


def visualize_forces(img, origin, ft, color=(255, 255, 0), force_scale=2e-2):
    """
    ft is a numpy array of shape (6,) in the camera frame
    the first three elements are the force vector
    we want to draw an arrow at the center of the image
    """
    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    force_ref_cam = ft[:3] @ ft_to_cam_rotation()

    camera_force = force_ref_cam.reshape(3, 1)  # (3,) -> (3, 1)
    located_camera_force = origin.reshape(3, 1) - force_scale * camera_force.reshape(3, 1)  # Subtract instead of adding
    origin_coords = cv2.projectPoints(
        origin.reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
    force_coords = cv2.projectPoints(
        located_camera_force, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]

    if np.isnan(force_coords).any() or np.isnan(origin_coords).any():
        return img

    pixel_mag = np.linalg.norm(force_coords - origin_coords)
    # if the force is too small or too large, don't draw it
    if pixel_mag < 2 or pixel_mag > 1000:
        return img

    # img = cv2.arrowedLine(img, (int(origin_coords[0]), int(origin_coords[1])),
                        #   (int(force_coords[0]), int(force_coords[1])), color, 4, tipLength=0.2)

    img = filled_arrowedLine(img, (int(origin_coords[0]), int(origin_coords[1])), (int(force_coords[0]), int(force_coords[1])), color, 4, tipLength=20)

    return img


def visualize_grip_force(img, grip_force, points, color=(0, 255, 0), force_scale=1e1):   
    if len(points) != 2:
        return img

    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    # draw arrowline for each fingertip pointing horizontally
    left_fingertip = points[0]
    right_fingertip = points[1]

    left_fingertip = cv2.projectPoints(
        points[0].reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]
    right_fingertip = cv2.projectPoints(
        points[1].reshape(3, 1), np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]

    # TODO: should we consider the force on gripper, or on object? currently the wrist force is in terms of force
    # on the gripper, and here the grip force is the force on target object
    grip_force = max(0.01, grip_force) # to avoid backwards arrows
    tiplength = 1 / grip_force
    img = cv2.arrowedLine(img, (int(left_fingertip[0]- force_scale * grip_force), int(left_fingertip[1])),
                          (int(left_fingertip[0] ), int(left_fingertip[1])), color, 4, tipLength=min(tiplength, 1))  # = 0.3)
    img = cv2.arrowedLine(img, (int(right_fingertip[0] + force_scale * grip_force), int(right_fingertip[1])),
                          (int(right_fingertip[0] ), int(right_fingertip[1])), color, 4, tipLength=min(tiplength, 1)) # = 0.3)
    return img


def visualize_datapoint(
            prompt, initial_data, final_data, config,
            pred=None, save_folder=None, viz_3d=None
        ):
    # visualize the datapoints and predictions
    prompt = prompt
    initial_rgb = initial_data['rgb']
    initial_depth = initial_data['depth']
    initial_state = initial_data['state']
    initial_left_fingertip = initial_data['left_fingertip']
    initial_right_fingertip = initial_data['right_fingertip']
    initial_ft = initial_data['ft']
    initial_grip_force = initial_data['grip_force']

    final_rgb = final_data['rgb']
    final_depth = final_data['depth']
    final_state = final_data['state']
    final_left_fingertip = final_data['left_fingertip']   # NOTE: future fingertip positions
    final_right_fingertip = final_data['right_fingertip'] # NOTE: future fingertip positions
    final_ft = final_data['ft']
    final_grip_force = final_data['grip_force']

    # print(' - initial rgb shape: ', initial_rgb.shape)
    # print(' - initial depth shape: ', initial_depth.shape)
    # print(' - final rgb shape: ', final_rgb.shape)
    # print(' - final depth shape: ', final_depth.shape)
    print(' - initial state: ', t2np(initial_state))
    print(' - final state: ', t2np(final_state))
    print(' - initial grip force: ', initial_grip_force)
    print(' - final grip force: ', final_grip_force)

    init_raw_ft = t2np(initial_ft)[:3] - EEF_PITCH_WEIGHT_OFFSET[:3]
    final_raw_ft = t2np(final_ft)[:3] - EEF_PITCH_WEIGHT_OFFSET[:3]
    print(' - initial raw ft: ', init_raw_ft, ' magnitude: ', np.linalg.norm(init_raw_ft))
    print(' - initial ft cam ref: ', init_raw_ft@ft_to_cam_rotation())
    print(' - final raw ft: ', final_raw_ft, ' magnitude: ', np.linalg.norm(final_raw_ft))
    print(' - final ft cam ref: ', final_raw_ft@ft_to_cam_rotation())

    # pos_dict = {"pitch": t2np(initial_ft)[4], "roll": t2np(initial_ft)[3]}
    # init_force_vec = ft_to_robot_frame(pos_dict, t2np(initial_ft)[:3])
    # print(' - initial ft in cam frame: ', init_force_vec)
    # pos_dict = {"pitch": t2np(initial_ft)[4], "roll": t2np(initial_ft)[3]}
    # final_force_vec = ft_to_robot_frame(pos_dict, t2np(final_ft)[:3])
    # print(' - final ft in bot frame: ', final_force_vec)
    # init_force_cam_vec = robot_frame_to_camera_frame(pos_dict, init_force_vec)
    # final_force_cam_vec = robot_frame_to_camera_frame(pos_dict, final_force_vec)

    initial_rgb, initial_depth = recover_rgbd(initial_rgb, initial_depth)
    final_rgb, final_depth = recover_rgbd(final_rgb, final_depth)

    print('gt point shape:', final_left_fingertip.cpu().numpy().shape)

    # save im for debugging
    save_img(initial_rgb, 'initial_rgb')
    save_img(final_rgb, 'final_rgb')
    # NOTE: this is transformed to the camera frame at initial frame    
    finger_tips_gt = [final_left_fingertip.cpu().numpy(), final_right_fingertip.cpu().numpy()]
    centroid = (finger_tips_gt[0] + finger_tips_gt[1])/2
    print('centroid:', centroid)
    # Visualize the fingertip positions
    initial_rgb_gt = visualize_points(initial_rgb.copy(), finger_tips_gt, colors=[(0, 0, 255), (0, 255, 0)])  # red = left, green = right
    initial_rgb_gt = visualize_forces(initial_rgb_gt, final_raw_ft, centroid[0])
    initial_rgb_gt = visualize_grip_force(initial_rgb_gt, t2float(final_grip_force), finger_tips_gt)

    ft_origin = (final_left_fingertip.cpu().numpy() + final_right_fingertip.cpu().numpy()) / 2
    # initial_rgb_gt = visualize_forces(initial_rgb_gt, final_ft, ft_origin, 

    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
        fig, axs = plt.subplots(2, 4)
        cls_img_gt, reg_img_gt = recover_pixel_space_represention(
            config, initial_data['cls_img'], initial_data['reg_img'])
        save_img(cls_img_gt, 'cls_img_gt')
        save_img(reg_img_gt, 'reg_img_gt')
    else:
        fig, axs = plt.subplots(2, 3)

    # plotting the initial rgb and depth images in the top row, and the final rgb and depth images in the bottom row
    # making the figure bigger
    fig.set_size_inches(18.5, 10.5)

    axs[0, 0].imshow(initial_rgb_gt)
    axs[0, 0].set_title('ground truth')
    axs[0, 1].imshow(initial_depth)
    axs[0, 1].set_title('initial depth')
    axs[1, 0].imshow(final_rgb)
    axs[1, 0].set_title('next keyframe')
    axs[1, 1].imshow(final_depth)
    axs[1, 1].set_title('final depth')
    axs[0, 2].set_title('prediction')
    axs[1, 2].set_title('final rgb')

    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:
        axs[0, 3].imshow(cls_img_gt)
        axs[0, 3].imshow(initial_rgb_gt, alpha=0.5)  # Overlay initial_rgb_gt with alpha transparency of 0.5
        axs[0, 3].set_title('cls_img')

        # Overlay initial_rgb_gt on reg_img
        axs[1, 3].imshow(reg_img_gt)
        axs[1, 3].imshow(initial_rgb_gt, alpha=0.5)  # Overlay initial_rgb_gt with alpha transparency of 0.5
        axs[1, 3].set_title('reg_img')

    if pred is not None:
        pred_force =  pred['force'].cpu().numpy()
        pred_grip_force =  t2float(pred['grip_force'])
        print('pred final force: ', pred_force)

        _has_fingertips_pred = True
        if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:

            cls_img_pred, reg_img_pred = recover_pixel_space_represention(config, pred['cls_img'], pred['reg_img'])
            save_img(cls_img_pred, 'cls_img_pred')
            save_img(reg_img_pred, 'reg_img_pred')

            if hasattr(config, 'PIXEL_SPACE_CENTROID') and config.PIXEL_SPACE_CENTROID:
                # classic pixel space representation for centroid
                centroid_pred = pixel_space_to_centroid(
                    config, cls_img_pred, reg_img_pred, method="local_max", threshold=0.002)
                if centroid_pred is None:
                    _has_fingertips_pred = False
                else:
                    if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
                        left_fingertip_pred, right_fingertip_pred = centroid_to_fingertips(centroid_pred, t2float(pred['width']), t2float(pred['yaw']))
                    else:
                        left_fingertip_pred, right_fingertip_pred = centroid_to_fingertips(centroid_pred, t2float(pred['width']))
            else:
                # classic pixel space representation for fingertips
                left_fingertip_pred, right_fingertip_pred = pixel_space_to_contacts(
                    config, cls_img_pred, reg_img_pred, method='local_max')

            # NOTE: When we are not detecting anything, we don't want to draw anything
            if _has_fingertips_pred:
                ft_origin = (left_fingertip_pred + right_fingertip_pred) /2
                # projecting the predicted fingertip positions onto cls_img_pred
                # NOTE: we kinda keep the normalization factor constant so we can compare what is actually being detected
                cls_img_pred_normalized = 1 - (cls_img_pred - np.min(cls_img_pred)) / 0.025 # (np.max(cls_img_pred) - np.min(cls_img_pred))
                cls_img_pred_rgb = cv2.applyColorMap((cls_img_pred_normalized*255).astype(np.uint8), cv2.COLORMAP_JET)

                # creating an image that mixes the predicted fingertip positions with the ground truth
                pred_overlay = cv2.addWeighted(cls_img_pred_rgb, 0.5, (initial_rgb*255).astype(np.uint8), 0.5, 0)
                pred_overlay = visualize_points(pred_overlay.copy(), [left_fingertip_pred, right_fingertip_pred],
                                                colors=[(255, 0, 255), (255, 0, 255)]) # both are purple
                print("drawing forces", left_fingertip_pred)
                pred_overlay = visualize_forces(pred_overlay, pred_force, ft_origin)
                pred_overlay = visualize_grip_force(pred_overlay, pred_grip_force, [left_fingertip_pred, right_fingertip_pred])
                axs[0, 2].imshow(pred_overlay)
                axs[1, 2].imshow(reg_img_pred, cmap='gray')
                axs[0, 2].set_title('pixel heatmap prediction')
                axs[1, 2].set_title('depth prediction')

                # computing the error
                left_error = 0.0
                right_error = 0.0
                if left_fingertip_pred is not None:
                    left_error = np.linalg.norm(left_fingertip_pred - finger_tips_gt[0])
                else:
                    print("left_fingertip_pred is None")
                if right_fingertip_pred is not None:
                    right_error = np.linalg.norm(right_fingertip_pred - finger_tips_gt[1])
                else:
                    print("right_fingertip_pred is None")

                # adding the error to the title
                axs[0, 2].set_title('prediction (left error: {:.4f} cm, right error: {:.4f} cm)'.format(left_error * 100, right_error * 100))

        else:
            # add a column to the figure for the prediction
            left_fingertip_pred, right_fingertip_pred = t2np(pred['left_fingertip']), t2np(pred['right_fingertip'])
            
            initial_rgb_pred = visualize_points(initial_rgb, [left_fingertip_pred, right_fingertip_pred],
                                                colors=[(0, 0, 255), (0, 255, 0)])
            print('left error (m): ', np.linalg.norm(left_fingertip_pred - final_left_fingertip.cpu().numpy()))
            print('right error (m): ', np.linalg.norm(right_fingertip_pred - final_right_fingertip.cpu().numpy()))
            left_error = np.linalg.norm(left_fingertip_pred - final_left_fingertip.cpu().numpy())
            right_error = np.linalg.norm(right_fingertip_pred - final_right_fingertip.cpu().numpy())
            # adding the error to the title
            axs[0, 2].set_title('prediction (left error: {:.4f} cm, right error: {:.4f} cm)'.format(left_error * 100, right_error * 100))
            axs[0, 2].imshow(initial_rgb_pred)
            axs[1, 2].imshow(final_rgb)
            axs[0, 2].set_title('prediction')
            axs[1, 2].set_title('final rgb')

        pred_force = t2np(pred['force'])
        pred_grip_force = t2np(pred['grip_force'])
        
        if viz_3d is not None and _has_fingertips_pred:
            final_ft_cam_vec = pred_force@ft_to_cam_rotation()
            centroid = (left_fingertip_pred + right_fingertip_pred) / 2
            viz_3d.publish_wrist_force(final_ft_cam_vec, centroid, is_curr=False)
            viz_3d.publish_grip_force_with_fingertips(
                pred_grip_force, [left_fingertip_pred, right_fingertip_pred])

        print('class probs: ', torch.softmax(pred['timestep'], dim=1))
        print('predicted class: ', np.argmax(pred['timestep'].cpu().numpy()))
        print('actual class: ', initial_data['timestep'])

    # setting overall title to the prompt
    fig.suptitle('prompt: ' + prompt[0], fontsize=16)

    # save plot for debugging
    if save_folder is not None:
        fig_index = create_file_index(save_folder)
        plt.savefig(os.path.join(save_folder, 'result_{}.png'.format(fig_index)))

    if viz_3d is not None:
        # This visualize the ground truth
        pcd = get_point_cloud(initial_rgb, initial_depth, Intrinsic640())

        assert len(finger_tips_gt) == 2
        # viz_3d.display(pcd, markers=finger_tips_gt)
        viz_3d.publish_pcd(pcd, invert_color=True)
        finger_tips_gt = [np.squeeze(a) for a in finger_tips_gt]
        final_ft_cam_vec = final_raw_ft@ft_to_cam_rotation()
        centroid = (finger_tips_gt[1] + finger_tips_gt[0]) / 2

        viz_3d.publish_wrist_force(final_ft_cam_vec, centroid, is_curr=True)
        viz_3d.publish_grip_force_with_fingertips(final_grip_force, finger_tips_gt, is_curr=True)

    directory = f'{os.path.expanduser("~")}/debug_imgs'
    plt.savefig(os.path.join(directory, 'result.png'))
    plt.show()


def visualize_prompt(img, prompt):
    """Visualizes the prompt on the image"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255) # color # Change to BLACK!!
    thickness = 2
    prompt_str = str(prompt)
    # Determine the size of the text
    text_size, _ = cv2.getTextSize(prompt_str, font, font_scale, thickness)
    # Calculate the coordinates to position the text in the left corner
    text_x = 10
    text_y = text_size[1] + 10
    # Add the text to the image
    cv2.putText(img, prompt_str, (text_x, text_y), font, font_scale, font_color, thickness)
    return img


def alphaMerge(small_foreground, background, top, left):
    """
    Puts a small BGRA picture in front of a larger BGR background.
    :param small_foreground: The overlay image. Must have 4 channels.
    :param background: The background. Must have 3 channels.
    :param top: Y position where to put the overlay.
    :param left: X position where to put the overlay.
    :return: a copy of the background with the overlay added.
    """
    result = background.copy()
    # From everything I read so far, it seems we need the alpha channel separately
    # so let's split the overlay image into its individual channels
    fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)
    # Make the range 0...1 instead of 0...255
    fg_a = fg_a / 255.0
    # Multiply the RGB channels with the alpha channel
    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

    # Work on a part of the background only
    height, width = small_foreground.shape[0], small_foreground.shape[1]
    part_of_bg = result[top:top + height, left:left + width, :]
    # Same procedure as before: split the individual channels
    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    # Merge them back with opposite of the alpha channel
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

    # Add the label and the part of the background
    cv2.add(label_rgb, part_of_bg, part_of_bg)
    # Replace a part of the background
    result[top:top + height, left:left + width, :] = part_of_bg
    return result


def filled_arrowedLine(img, pt1, pt2, color, thickness=1, line_type=8, tipLength=20):
    # Convert points to numpy arrays
    pt1 = np.array(pt1, dtype=np.float32)
    pt2 = np.array(pt2, dtype=np.float32)
    
    # Draw main line of the arrow
    cv2.line(img, tuple(pt1.astype(int)), tuple(pt2.astype(int)), color, thickness, line_type)
    
    # Calculate the angle of the arrow
    angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    
    # Calculate the tip size based on the length of the arrow and the tipLength factor
    # tipSize = np.linalg.norm(pt1 - pt2) * tipLength
    tipSize = tipLength

    # # moving pt2 to the tip of the arrow
    pt2_long = np.array([pt2[0] - tipSize * np.cos(angle),
                    pt2[1] - tipSize * np.sin(angle)], dtype=np.float32)

    # Calculate points for the arrowhead
    p1 = np.array([pt2_long[0] + tipSize * np.cos(angle + np.pi/12), 
                   pt2_long[1] + tipSize * np.sin(angle + np.pi/12)], dtype=np.int32)
    p2 = np.array([pt2_long[0] + tipSize * np.cos(angle - np.pi/12), 
                   pt2_long[1] + tipSize * np.sin(angle - np.pi/12)], dtype=np.int32)
    
    # Draw the filled arrowhead
    cv2.fillPoly(img, np.array([[tuple(pt2_long.astype(int)), p1, p2]], dtype=np.int32), color)

    return img
