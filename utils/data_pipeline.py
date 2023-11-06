#!/usr/bin/env python3

from utils.realsense_utils import Intrinsic640, get_point_cloud, display_point_cloud
import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from prediction.models import *
from prediction.deep_fusion import DeepFusion
from prediction.classifier_free_guidance import *
from prediction.grip_force_model import GripForceMLP
from robot.robot_utils import *
from utils.transform import *

from stretch_remote.robot_utils import read_robot_status
from skimage.feature import peak_local_max

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import shutil
from pathlib import Path
import ast
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def timestep_to_subgoal_text(timestep: int, text_prompt=None) -> str:
    """
    These text are appended to the prompt to indicate the subgoal.
     - apple: approach, grasp, lift, done
     - medicine bottle: approach, grasp, lift, done
     - paperclip: approach, grasp, lift, done
     - apple: approach, grasp, lift, done
     - cup: approach, grasp, lift, done
     - hand sanitizer: approach, grasp, lift, done
     - drawer: approach, grasp, pull, ungrasp, done
     - light switch: approach, push, done
     - trash: approach, ungrasp, done
     - hand: approach, ungrasp, done
    """
    # TODO: to decide, this seems hacky
    # This is the default method, we will further refine it
    if text_prompt is None:
        if timestep == 0: # 1-2 approach and pregrasp
          return ", approach"
        elif timestep == 1: # 2-3 grasp
            return ", grasp"
        elif timestep == 2:
            return ", lift"
        else:
            return ", done"

    # else use this new task specific method
    if any(i in text_prompt for i in ["pick", "grab", "take"]):
        seq = ["approach", "grasp", "lift"]
    elif any(i in text_prompt for i in ["place", "put", "drop"]):
        seq = ["approach", "ungrasp"]
    elif any(i in text_prompt for i in ["switch", "push", "flip", "close"]):
        seq = ["approach", "push"]
    elif any(i in text_prompt for i in ["open", "pull"]):
        seq = ["approach", "grasp", "pull", "ungrasp"]
    else:
        seq = []
    if timestep > len(seq) - 1:
        return ", done"
    else:
        return ", " + seq[timestep]

def load_state(config, data_point):
    with open(data_point, 'r') as f:
        state_data = torch.tensor([])
        robot_state = read_robot_status(ast.literal_eval(f.read()))

        for state in config.ROBOT_STATES:
            if state in robot_state:
                state_data = torch.cat((state_data, torch.tensor([robot_state[state]])), dim=0)
            else:
                print('state not found: ', state)
                state_data = torch.cat((state_data, torch.tensor([0.0])), dim=0)
    
    return state_data

def filter_data(rgb_paths):
    keyframe_path = os.path.join(str(Path(rgb_paths['final'][0]).parent.parent), 'keyframe_list.npz')
    print('rgb_paths: ', rgb_paths)
    print('initial rgb', rgb_paths['initial'])
    print('final rgb', rgb_paths['final'])

    delete = input('Delete this data point? (y/n): ')

    while delete not in ['y', 'n']:
        delete = input('Delete this data point? (y/n): ')
    if delete == 'y':
        # delete the data point from keyframe_list.npz
        keyframe_list = np.load(keyframe_path, allow_pickle=True)
        keyframe_index_list = keyframe_list['keyframe_index_list']
        keyframe_step_list = keyframe_list['keyframe_step_list']

        initial_or_final = input('Delete initial or final? (press 1 for initial, press 2 for final)')

        while initial_or_final not in ['1', '2']:
            initial_or_final = input('Delete initial or final? (press 1 for initial, press 2 for final)')
        if initial_or_final == '1':
            delete_index = np.where(keyframe_index_list == int(rgb_paths['initial'][0].split('/')[-1].split('.')[0]))
        elif initial_or_final == '2':
            delete_index = np.where(keyframe_index_list == int(rgb_paths['final'][0].split('/')[-1].split('.')[0]))
        else:
            print('invalid input')

        print('old keyframe_index_list: ', keyframe_index_list.shape)
        print('old keyframe_step_list: ', keyframe_step_list.shape)

        # delete the element of the keyframe_index_list that has the same value as the name of the initial_rgb (e.g. 13.png)
        print(f'DELETING index {delete_index} from {keyframe_path}')
        keyframe_step_list = np.delete(keyframe_step_list, delete_index)
        keyframe_index_list = np.delete(keyframe_index_list, delete_index)

        print('new keyframe_index_list: ', keyframe_index_list.shape)
        print('new keyframe_step_list: ', keyframe_step_list.shape)

        input('Save changes? (press enter to save, press ctrl+c to cancel)')
        np.savez(keyframe_path, keyframe_index_list=keyframe_index_list, keyframe_step_list=keyframe_step_list)
        print('new keyframe_list saved to ', keyframe_path)

def normalize_gripper_pos(gripper_val):
    # map the gripper values from [-100, 60] to [0, 1]
    gripper_val = (gripper_val + 100) / 160.0
    gripper_val[gripper_val < 0] = 0
    gripper_val[gripper_val > 1] = 1
    gripper_val[torch.isnan(gripper_val)] = 0
    return gripper_val

def normalize_gripper_effort(gripper_effort):
    return gripper_effort / 30.0

def preprocess_prompt(config, prompt, tokenizer):
    # Add padding to the text input so it can be processed by the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.MULTIMODAL_HEAD == 'classifier-free-guidance':
        # converting tuple to list. cfg stuff handles tokenization
        processed_prompt = list(prompt)
    elif config.TEXT_MODEL in ['t5-small', 't5-base', 't5-large']:
        processed_prompt = tokenizer.batch_encode_plus(
        prompt,
        padding='max_length',
        max_length=32,
        return_tensors='pt',
        truncation=True
        )["input_ids"].to(device)
    elif config.TEXT_MODEL.split('-')[0] == 'clip':
        processed_prompt = tokenizer(prompt).to(device)
    elif config.TEXT_MODEL.split('-')[0] == 'bert':
        # processed_prompt = tokenizer(prompt, padding='max_length',
        #       max_length=32, return_tensors='pt', truncation=True)["input_ids"].to(device)
        processed_prompt = tokenizer(prompt, return_tensors="pt").to(device)
        
    return processed_prompt

def convert_image_to_rgb(image):
        return image.convert("RGB")

def preprocess_rgbd(config, rgb_image, depth_image):
    # process the raw images from the realsense for the model
    if 'crop' in config.TRANSFORM:
        # cropping to a square in the middle of the image (480, 640) -> (480, 480)
        rgb_image = rgb_image[:, 80:560, :]
        depth_image = depth_image[:, 80:560]     

    if config.IMAGE_MODEL.startswith('clip'):
        clip_transforms = Compose([
                            Resize(config.IMAGE_SIZE, interpolation=BICUBIC),
                            CenterCrop(config.IMAGE_SIZE),
                            convert_image_to_rgb,
                            ToTensor(),
                            Normalize((0.48145466, 0.4578275, 0.40821073),
                                      (0.26862954, 0.26130258, 0.27577711)),
                            ])
        
        # converting to PIL image
        rgb_image = Image.fromarray(rgb_image)
        rgb_image = clip_transforms(rgb_image)
    else:
        rgb_image = rgb_image.astype(np.float32)
        rgb_image = cv2.resize(rgb_image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0

    depth_image = depth_image.astype(np.float32)
    depth_image = cv2.resize(depth_image, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                             interpolation=cv2.INTER_NEAREST)
    depth_image = torch.from_numpy(depth_image).unsqueeze(0).float() / 65535.0

    if 'jitter' in config.TRANSFORM:
        # torchvision color jitter
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        rgb_image = color_jitter(rgb_image)

    if 'invert' in config.TRANSFORM:
        # color inversion 50% of the time
        if random.random() > 0.5:
            rgb_image = 1 - rgb_image

    return rgb_image, depth_image

def pixel_space_representation(config, contacts) -> (np.ndarray, np.ndarray):
    """
    This function takes in the 3d contact points of the image 2, config.IMAGE_SIZE
    :return:    the 2d pixel space ground truth
                classification labels for xy, and regression labels for z depth
    """
    # TODO: output normalized to float32/64 depends on default, also cls sum should be 1?
    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    if 'crop' in config.TRANSFORM:
        x_dim = 480
        intr.ppx -= 80  # since crop will change the ppx (original_width - new_width) / 2
    else:
        x_dim = 640

    cls_img = np.zeros((480, x_dim), np.float32)
    reg_img = np.zeros((480, x_dim), np.float32)

    for point in contacts:
        # NOTE: point might be -z since we predicting a future action
        # thus we will make future contact points that is behind the camera to be at least
        # the in view of the camera
        if point[2] < 0:
            if hasattr(config, 'CLIP_MIN_Z_AXIS') and config.CLIP_MIN_Z_AXIS:
                point[2] = config.CLIP_MIN_Z_AXIS
            else:
                # With FOV of 1.01 on h, z = 5/tan(fov/2), with 5cm is the offset of the
                # fingertip to the camera on the y-axis, this is the minimum z-axis
                point[2] = 0.09 # 9cm

        point2d = cv2.projectPoints(
            point, np.zeros((3, 1)), np.zeros((3, 1)), cam_mat, cam_dist)[0][0][0]

        # NOTE: currently we are doing a image space clipping since
        # the point2d coord can be out of bounds
        # print('point2d: ', point2d)
        # TODO: return None instead of clipping when the point is out of bounds
        # we'll then need to handle the None when the data is loaded

        int_point2d = (int(np.clip(point2d[0], 0, x_dim -1)),
                        int(np.clip(point2d[1], 0, 480 - 1)))
        # add z-distance text to the image
        z_dist = np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        z_dist /= intr.depth_scale

        # set pixel value to 1
        if hasattr(config, 'CLS_POINT_RADIUS') and config.CLS_POINT_RADIUS:
            radius = config.CLS_POINT_RADIUS
            # draw a filled circle in the image at the point2d with the radius
            cv2.circle(cls_img, (int_point2d[0], int_point2d[1]), radius, (1), -1)
            cv2.circle(reg_img, (int_point2d[0], int_point2d[1]), radius, (z_dist), -1)
        else:
            cls_img[int_point2d[1], int_point2d[0]] = 1
            reg_img[int_point2d[1], int_point2d[0]] = z_dist

    cls_img = cv2.resize(cls_img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    reg_img = cv2.resize(reg_img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    # cls with 1s and reg with z-distances
    return cls_img, reg_img

def preprocess_pixel_space(cls_img, reg_img):
    # normalize imgs to 0-1
    cls_img = torch.from_numpy(cls_img).unsqueeze(0).float()
    reg_img = torch.from_numpy(reg_img).unsqueeze(0).float() / 65535.0 # 2^16
    return cls_img, reg_img

def recover_pixel_space_represention(config, cls_img, reg_img):
    cls_img = cls_img.detach().squeeze(0).cpu().numpy()[0]
    reg_img = reg_img.detach().squeeze(0).cpu().numpy()[0]
    reg_img = reg_img * 65535.0
    x_dim = 480 if 'crop' in config.TRANSFORM else 640

    # resize back to 480x640
    cls_img = cv2.resize(cls_img, (x_dim, 480), interpolation=cv2.INTER_NEAREST)
    reg_img = cv2.resize(reg_img, (x_dim, 480), interpolation=cv2.INTER_NEAREST)

    # return contact
    return cls_img, reg_img

def centroid_to_fingertips(centroid, width, yaw=0):
    """
    takes in a centroid, width, and yaw, output a left and right fingertips
    """
    left_fingertip = centroid - width/2 * np.array([np.cos(yaw), 0, np.sin(yaw)])
    right_fingertip = centroid + width/2 * np.array([np.cos(yaw), 0, np.sin(yaw)])    
    return left_fingertip, right_fingertip

def fingertips_to_centroid_width_yaw(left_fingertip, right_fingertip):
    """convert from fingertips to centroid, yaw, width representation"""
    yaw = np.arctan2(right_fingertip[2] - left_fingertip[2],
                     right_fingertip[0] - left_fingertip[0])
    width = np.linalg.norm(right_fingertip - left_fingertip)
    centroid = (left_fingertip + right_fingertip) / 2
    return centroid, width, yaw

def pixel_space_to_centroid(config, cls_img, reg_img, threshold=0.005, method="global_max",
                            avg_depth_within_radius=None):
    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()
    # get height width form cls_img
    h, w = cls_img.shape

    if method == "global_max":
        # get coord of max value in the cls_img
        max_index = np.unravel_index(np.argmax(cls_img), cls_img.shape)
        # The below impl is the same as pixel_space_to_contacts()
        if cls_img[max_index] < threshold:
            return None
        y, x = max_index
    elif method == "local_max":
        # Find local peaks in the cls_img
        coordinates = peak_local_max(cls_img,
                                    num_peaks=1,
                                    threshold_abs=threshold,
                                    min_distance=20,
                                    exclude_border=False)
        if len(coordinates) == 0:
            return None
        y, x = coordinates[0]

    if avg_depth_within_radius is not None:
        # get the reg_img value at a radius around y, x and average it.
        # make sure y and x are within bounds
        start_x = max(0, x - avg_depth_within_radius)
        start_y = max(0, y - avg_depth_within_radius)
        end_x = min(w, x + avg_depth_within_radius)
        end_y = min(h, y + avg_depth_within_radius)
        region = reg_img[start_y:end_y, start_x:end_x]
        ray_z_dist = np.mean(region) * intr.depth_scale
    else:
        ray_z_dist = reg_img[y, x] * intr.depth_scale

    # Construct a 2D point in normalized image coordinates
    point2d = np.array([[[
        (x - cam_mat[0, 2]) / cam_mat[0, 0],
        (y - cam_mat[1, 2]) / cam_mat[1, 1]]]], dtype=np.float32)
    xy_dist = np.sqrt(point2d[0, 0, 0]**2 + point2d[0, 0, 1]**2)
    z_dist = np.sqrt(ray_z_dist**2/(1 + xy_dist**2))
    # NOTE: for now treat image as undistorted
    # point2d = cv2.undistortPoints(point2d, cam_mat, cam_dist)
    # Project the 2D point to 3D. Here we make use of the known depth (z-coordinate)
    point3d = np.array([point2d[0, 0, 0] * z_dist, point2d[0, 0, 1] * z_dist, z_dist])
    return point3d

def pixel_space_to_contacts(config, cls_img, reg_img,
                            method='threshold', filter_threshold=0.005):
    """
    This converts the pixel space representation to 3d contacts
    NOTE: the cls and reg img is in or 640x480 or 480x480, which is used
          after the recover_pixel_space_represention() function
    """
    intr =  Intrinsic640()
    cam_mat = intr.cam_mat()
    cam_dist = intr.cam_dist()

    # sum up all the pixel values in the cls_img
    # print(' [DEBUG] cls_img sum: ', np.sum(cls_img))
    if method == 'threshold':  
        # Threshold the image to make it binary
        _, thresholded = cv2.threshold(cls_img, filter_threshold, 255, cv2.THRESH_BINARY)
        thresholded = thresholded.astype('uint8')
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            # Find the moments of the contour which can be used to find the center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # Calculate x, y coordinate of center
                cX = int(M["m01"] / M["m00"])
                cY = int(M["m10"] / M["m00"])
                centers.append(np.array([cX, cY]))
        if len(centers) == 0:
            return None, None
        else:
            coordinates = np.concatenate(centers).reshape(len(centers), -1)

    elif method == 'local_max':
        # Find local peaks in the cls_img
        coordinates = peak_local_max(cls_img,
                                    num_peaks=4, # fix to 6 contacts
                                    threshold_abs=filter_threshold,
                                    min_distance=20,
                                    exclude_border=False)
    # NOTE: debug show coordinates on an image
    # plt.imshow(cls_img, cmap='gray')
    # plt.plot(coordinates[:, 1], coordinates[:, 0], 'r+', markersize=10)
    # plt.show()
    # print("number of candidates", len(coordinates))

    contacts = []
    for i in range(len(coordinates)):
        y, x = coordinates[i]
        ray_z_dist = reg_img[y, x] * intr.depth_scale
        # Construct a 2D point in normalized image coordinates
        point2d = np.array([[[
            (x - cam_mat[0, 2]) / cam_mat[0, 0],
            (y - cam_mat[1, 2]) / cam_mat[1, 1]]]], dtype=np.float32)
        xy_dist = np.sqrt(point2d[0, 0, 0]**2 + point2d[0, 0, 1]**2)
        z_dist = np.sqrt(ray_z_dist**2/(1 + xy_dist**2))
        # NOTE: for now treat image as undistorted
        # point2d = cv2.undistortPoints(point2d, cam_mat, cam_dist)
        # Project the 2D point to 3D. Here we make use of the known depth (z-coordinate)
        point3d = np.array([point2d[0, 0, 0] * z_dist, point2d[0, 0, 1] * z_dist, z_dist])
        contacts.append(point3d)

    if len(contacts) == 0:
        return None, None
    elif len(contacts) == 1: # if only one contact point
        return contacts[0], contacts[0]
    elif len(contacts) == 2:
        left_fingertip = contacts[0]
        right_fingertip = contacts[1]
    elif len(contacts) > 2:
        # find with pair with the lowest pairwise distances between all the 
        min_dist = 1000000
        for i in range(len(contacts)):
            for j in range(i + 1, len(contacts)):
                dist = np.linalg.norm(contacts[i] - contacts[j])
                if dist < min_dist:
                    min_dist = dist
                    left_fingertip = contacts[i]
                    right_fingertip = contacts[j]
    else:
        raise ValueError('invalid number of contacts')

    #check if distance between contacts is too far
    if np.linalg.norm(left_fingertip - right_fingertip) > 0.18:
        return left_fingertip, left_fingertip

    # make sure the left fingertip is the element with the smallest x value
    temp = left_fingertip
    left_fingertip = left_fingertip if left_fingertip[0] < right_fingertip[0] else right_fingertip
    right_fingertip = right_fingertip if temp[0] < right_fingertip[0] else temp
    return left_fingertip, right_fingertip

def postprocess_output(config, output, stage='train'):
    # convert to dict if not already
    if isinstance(output, dict):
        return output

    if hasattr(config, 'PIXEL_SPACE_OUTPUT') and config.PIXEL_SPACE_OUTPUT:       
        img_t_size = config.IMAGE_SIZE**2
        # taking softmax of cls_img over the pixels (it's a tensor of shape (batch_size, 1, 224, 224))
        cls_output = output[:, 0:img_t_size]
        if stage == 'metrics':
            cls_output= torch.nn.functional.softmax(cls_output, dim=1)
        # cls_output = cls_output.view(-1, 1, 196, 256)
        # cls_output = cls_output.permute(0, 3, 1, 4, 2, 5).contiguous()
        cls_output = cls_output.view(-1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
        output_dict = {
                # 'cls_img': output[:, 0:img_t_size].reshape(-1, 1,  config.IMAGE_SIZE, config.IMAGE_SIZE),
                # using view instead
                'cls_img': cls_output,
                'reg_img': output[:, img_t_size:2*img_t_size].reshape(-1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE),
                'force': output[:, 2*img_t_size:2*img_t_size+3],
                'grip_force': output[:, 2*img_t_size+3],
                'timestep': output[:, 2*img_t_size+4:2*img_t_size+4 + config.NUM_TIMESTEPS]
            }
        if hasattr(config, 'LAMBDA_WIDTH') and config.LAMBDA_WIDTH:
            output_dict['width'] = output[:, 2*img_t_size+4 + config.NUM_TIMESTEPS]

        if hasattr(config, 'LAMBDA_YAW') and config.LAMBDA_YAW:
            output_dict['yaw'] = output[:, 2*img_t_size+4 + config.NUM_TIMESTEPS + 1]
    else:
        # print output shape
        output_dict = {
                'left_fingertip': output[:, 0:3],
                'right_fingertip': output[:, 3:6],
                'force': output[:, 6:9],
                'grip_force': output[:, 9]
            }
        if hasattr(config, 'NUM_TIMESTEPS') and config.NUM_TIMESTEPS:       
            output_dict['timestep'] = output[:, 10:10 + config.NUM_TIMESTEPS]
    return output_dict

def recover_rgbd(rgb_image, depth_image):
    """convert tensor to numpy array and reshape to 640x480"""
    rgb_image = rgb_image.cpu().numpy().squeeze()
    rgb_image = np.transpose(rgb_image, (1, 2, 0))
    depth_image = depth_image.cpu().numpy().squeeze()

    # rgb to bgr
    rgb_image = rgb_image[:, :, ::-1]

    # reshaping to 640x480
    rgb_image = cv2.resize(rgb_image, (640, 480))
    depth_image = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_NEAREST)
    return rgb_image, depth_image
