#!/usr/bin/env python3

from typing import List, Optional
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
import sys
from utils.config_utils import *
from utils.realsense_utils import *
from robot.robot_utils import *
from utils.aruco_detect import ArucoPoseEstimator, find_contact_markers
from utils.transform import get_transformed_fingertips
from utils.pred_utils import *
from utils.data_pipeline import *
from utils.data_aug import *
from utils.visualizer import *

import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class ActAffData(data.Dataset):
    def __init__(self, folder='/data/raw/', stage='raw', shuffle=True):
        self.config, self.args = parse_config_args()
        self.stage = stage

        # to handle old configs
        if type(folder) == str:
            self.root = [os.path.join(os.getcwd(), folder)]
        
        elif type(folder) == list:
            self.root = folder

        print('Loading data from', self.root)

        self.grip_force_model = load_grip_force_model()
        self.cam_intr = Intrinsic640()

        if self.args.bipartite:
            self.dataset = self.get_data_bipartite(self.root, shuffle=shuffle)
        else:
            self.dataset = self.get_data(self.root, shuffle=shuffle)

        # This removes the data point with non-viewable target (aka fingertip is out of view)
        # this only considers the x and y axis
        print('\n Total data points:', len(self.dataset))
        if hasattr(self.config, 'REMOVE_NON_VIEWABLE_TARGET') and \
            self.config.REMOVE_NON_VIEWABLE_TARGET and \
            self.args.ignore_prefilter is False:
            ## this prefiltering step removes data points with non-viewable target
            x_dim = 480 if 'crop' in self.config.TRANSFORM else 640
            total_index = len(self.dataset)
            index = 0
            while index < total_index:
                fingertips, timestep_idx = self.__getitem__(index, prefilter_return=True)
                is_out = 0

                # NOTE: we only filter 1st Subgoal
                if timestep_idx != 0:
                    index += 1
                    continue

                for finger in fingertips:
                    point2d = cv2.projectPoints(finger, np.zeros((3, 1)), np.zeros((3, 1)),
                        self.cam_intr.cam_mat(), self.cam_intr.cam_dist())[0][0][0]
                    if point2d[0] < 0 or point2d[0] > x_dim or point2d[1] < 0 or point2d[1] > 480:
                        is_out += 1
                if is_out >= self.config.REMOVE_NON_VIEWABLE_TARGET:
                    del self.dataset[index]
                    total_index -= 1
                    # print('remove sample with timestep_idx', timestep_idx)
                else:
                    index += 1
        print(' \n After Remove non viewable targets, Total data points:', len(self.dataset))
        print(f' \n Loaded {len(self.dataset)} data points')

    def __getitem__(self, index, prefilter_return=False):
        # data_point = self.dataset[index]
        prompt_path, initial_paths, final_paths = self.dataset[index]

        with open(prompt_path, 'r') as file:
            prompt = file.read()

        # Load the data from the paths
        initial_data = {}
        final_data = {}

        # State
        initial_data['state'] = load_state(self.config, initial_paths['state'])
        final_data['state'] = load_state(self.config, final_paths['state'])

        # TimeStep
        # make timestep a one-hot tensor with size self.config.NUM_TIMESTEPS
        parent_folder = Path(final_paths['ft']).parent.parent
        timestep_idx = int(str(parent_folder).split('/')[-1].split('_')[-3]) - 1 # file names must end in frame_x_y_z, where x and y are the initial and final timesteps
        timestep = torch.zeros(self.config.NUM_TIMESTEPS).scatter_(0, torch.tensor(timestep_idx), 1) # shape (NUM_TIMESTEPS,), one-hot tensor
        # timestep = torch.tensor(timestep)
        initial_data['timestep'] = timestep

        # fingertips
        initial_data['left_fingertip'] = np.load(initial_paths['fingertips'], allow_pickle=True)['left'] # aruco
        initial_data['right_fingertip'] = np.load(initial_paths['fingertips'], allow_pickle=True)['right'] # aruco
        initial_data['left_fingertip'], initial_data['right_fingertip'] = find_contact_markers(initial_data['left_fingertip'], initial_data['right_fingertip'])

        final_data['left_fingertip'] = np.load(final_paths['fingertips'], allow_pickle=True)['left'] # aruco
        final_data['right_fingertip'] = np.load(final_paths['fingertips'], allow_pickle=True)['right'] # aruco
        final_data['left_fingertip'], final_data['right_fingertip'] = find_contact_markers(final_data['left_fingertip'], final_data['right_fingertip'])

        # This is used only to speed up the data iteration process without reading the entire dataset and manage image
        # pre post processing, Used in REMOVE_NON_VIEWABLE_TARGET config
        if prefilter_return:
            return get_transformed_fingertips(
                initial_data['state'], final_data['state'],
                final_data['left_fingertip'], final_data['right_fingertip']), timestep_idx

        initial_data['rgb'] = cv2.imread(initial_paths['rgb'], cv2.IMREAD_COLOR)
        initial_data['depth'] = cv2.imread(initial_paths['depth'], cv2.IMREAD_ANYDEPTH)

        ###############################################################################################

        # uniform distribution of value between -80 and 80 pixels
        if self.args.random_trans:
            x_trans = np.random.randint(-80, 80)
            y_trans = np.random.randint(-80, 80)
            initial_data['rgb'], initial_data['depth'], translated_img_transform = \
                translate_imgs(initial_data['rgb'], initial_data['depth'], x_trans, y_trans, self.cam_intr)
        else:
            translated_img_transform = None

        final_data['rgb'] = cv2.imread(final_paths['rgb'], cv2.IMREAD_COLOR)
        final_data['depth'] = cv2.imread(final_paths['depth'], cv2.IMREAD_ANYDEPTH)

        initial_data['rgb'], initial_data['depth'] = preprocess_rgbd(self.config, initial_data['rgb'] , initial_data['depth'])
        final_data['rgb'], final_data['depth'] = preprocess_rgbd(self.config, final_data['rgb'] , final_data['depth'])

        # force and torque
        initial_data['ft'] = np.load(initial_paths['ft'])
        final_data['ft'] = np.load(final_paths['ft'])

        # TODO: this will update the raw final data left and right fingertip data
        # should rename this to transformed left and right fingertip data
        final_data['left_fingertip'], final_data['right_fingertip'] = \
            get_transformed_fingertips(
                initial_data['state'], final_data['state'],
                final_data['left_fingertip'], final_data['right_fingertip'],
                translated_img_transform,
            )
        final_data['centroid'], final_data['width'], final_data['yaw'] = fingertips_to_centroid_width_yaw(
            final_data['left_fingertip'], final_data['right_fingertip'])

        l_contact, r_contact = final_data['left_fingertip'], final_data['right_fingertip']

        if hasattr(self.config, 'PIXEL_SPACE_OUTPUT') and self.config.PIXEL_SPACE_OUTPUT:
            if hasattr(self.config, 'PIXEL_SPACE_CENTROID') and self.config.PIXEL_SPACE_CENTROID:
                target = [final_data['centroid']]
            else:
                target = [l_contact, r_contact]
            cls_img, reg_img = pixel_space_representation(self.config, target)
            initial_data['cls_img'], initial_data['reg_img'] = preprocess_pixel_space(cls_img, reg_img)

        # Loading force-torque data and subtracting offset for both initial and final data
        ft_offset = np.load(os.path.join(parent_folder, 'ft_calibration.npy'))

        initial_data['ft'] = initial_data['ft'] - ft_offset
        initial_data['ft'] = torch.from_numpy(initial_data['ft'])

        final_data['ft'] = final_data['ft'] - ft_offset
        final_data['ft'] = torch.from_numpy(final_data['ft'])

        rgb_paths = { # we load these to access file names in the loader for filtering
            'initial': initial_paths['rgb'],
            'final': final_paths['rgb']
        }

        with open(initial_paths['state'], 'r') as f:
            pos_dict_initial = (read_robot_status(ast.literal_eval(f.read())))
        with open(final_paths['state'], 'r') as f:
            pos_dict_final = (read_robot_status(ast.literal_eval(f.read())))

        initial_data['grip_force'] = run_grip_force_model(self.grip_force_model, pos_dict_initial, initial_data['left_fingertip'], initial_data['right_fingertip'])
        final_data['grip_force'] = run_grip_force_model(self.grip_force_model, pos_dict_final, final_data['left_fingertip'], final_data['right_fingertip'])

        # NOTE: postappend the current subgoal to the prompt
        if hasattr(self.config, 'SUBGOAL_TEXT') and self.config.SUBGOAL_TEXT:
            if self.config.SUBGOAL_TEXT == 'named_action':
                prompt += timestep_to_subgoal_text(timestep_idx, prompt)
            else:
                prompt += timestep_to_subgoal_text(timestep_idx)

        return prompt, initial_data, final_data, rgb_paths

    def __len__(self):
        return len(self.dataset)

    ########################################################################

    def get_data(self, root_dir: List, shuffle=True):
        """ 
        Get data from the root directory and return a list dictionaries with the file paths for the prompt, initial data, and final data
        : Return datatype as such    
        (
            prompt_path,
            
            initial_paths = {
                'rgb': rgb1[1],
                'depth': depth1[1],
                'state': state1[1],
                'fingertips': fingertips1[1],
                'ft': ft1[1],
            },
            
            final_paths = {
                'rgb': rgb2[1],         # at frame_final rgb image
                'depth': depth2[1],     # at frame_final depth image
                'state': state2[1],     # at frame_final robot state
                'fingertips': fingertips2[1],      # ref to final frame finger tips location -> 
                                                    # ref to frame_initial, coordinate of the future finger tips
                'ft': ft2[1],
            }
        )   
        """
        rgb_names = []
        depth_names = []
        state_names = []
        fingertips_names = []
        ft_names = []
        dataset = []
        data_parents = []

        # crawling the directory to sort the data modalities 
        # also saving the data parents (one folder per run of capture_data())
        for folder in root_dir:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    # saving (timestep, path)
                    parent = root.split('/')[-1]
                    if parent == 'rgb' and file.endswith('.png'):
                        rgb_names.append((int(file[:-4]), os.path.join(root, file)))
                    elif parent == 'depth' and file.endswith('.png'):
                        depth_names.append((int(file[:-4]), os.path.join(root, file)))
                    elif parent == 'ft' and file.endswith('.npy'):
                        ft_names.append((int(file[:-4]), os.path.join(root, file)))
                    elif parent == 'state' and file.endswith('.txt'):
                        state_names.append((int(file[:-4]), os.path.join(root, file)))
                    elif parent == 'fingertips' and file.endswith('.npz'):
                        fingertips_names.append((int(file[:-4]), os.path.join(root, file)))

                    potential_keyframe_path = os.path.join(str(Path(os.path.join(root, file)).parent.parent), 'keyframe_list.npz')
                    potential_ft_calib_path = os.path.join(str(Path(os.path.join(root, file)).parent.parent), 'ft_calibration.npy')
                    if os.path.exists(potential_keyframe_path) and os.path.exists(potential_ft_calib_path): # if the keyframe list and ft calibration file exist
                        data_parents.append(str(Path(os.path.join(root, file)).parent.parent))

        # sorting the data modalities by index
        for name in [rgb_names, depth_names, state_names, fingertips_names, ft_names]:
            name.sort(key=lambda x: x[0])

        data_parents = list(set(data_parents)) # deduplication
        print('Number of folders: ', len(data_parents))

        # getting the keyframe list for each episode
        for data_folder in data_parents:
            keyframe_index_list = np.load(os.path.join(data_folder, 'keyframe_list.npz'), allow_pickle=True)['keyframe_index_list']
            prompt_path = os.path.join(data_folder, 'prompt.txt')

            # with open(prompt_path, 'r') as file:
            #     prompt = file.read()

            '''
            The dataset contains pairs of (rgb, depth, prompt, state, fingertips, ft), as well as the prompt.
            The pairs are adjacent in the keyframe list, and should come from the same folder
            Data that is not in the keyframe list is ignored
            '''
            for i, index in enumerate(keyframe_index_list[:-1]):
                next_index = keyframe_index_list[i + 1]

                # finding the first item in the list that matches the index and is in the same folder (Path(rgb_names[0][1]).parent.parent == data_folder, for example)
                rgb1 = next((item for item in rgb_names if item[0] == index and str(Path(item[1]).parent.parent) == data_folder), None)
                depth1 = next((item for item in depth_names if item[0] == index and str(Path(item[1]).parent.parent) == data_folder), None)
                state1 = next((item for item in state_names if item[0] == index and str(Path(item[1]).parent.parent) == data_folder), None)
                fingertips1 = next((item for item in fingertips_names if item[0] == index and str(Path(item[1]).parent.parent) == data_folder), None)
                ft1 = next((item for item in ft_names if item[0] == index and str(Path(item[1]).parent.parent) == data_folder), None)
                # finding the first item in the list that matches the next index
                rgb2 = next((item for item in rgb_names if item[0] == next_index and str(Path(item[1]).parent.parent) == data_folder), None)
                depth2 = next((item for item in depth_names if item[0] == next_index and str(Path(item[1]).parent.parent) == data_folder), None)
                state2 = next((item for item in state_names if item[0] == next_index and str(Path(item[1]).parent.parent) == data_folder), None)
                fingertips2 = next((item for item in fingertips_names if item[0] == next_index and str(Path(item[1]).parent.parent) == data_folder), None)
                ft2 = next((item for item in ft_names if item[0] == next_index and str(Path(item[1]).parent.parent) == data_folder), None)

                if all([prompt_path, rgb1, depth1, state1, rgb2, depth2, fingertips2, ft2]): # if all the items are found
                    # each data point is a dictionary with keys 'prompt', 'initial', 'final'
                    # initial and final are dictionaries with keys 'rgb', 'depth', 'state', 'fingertips', 'ft'
                    # for example, initial_paths['rgb'] is the rgb image of the initial state

                    initial_paths = {
                        'rgb': rgb1[1],
                        'depth': depth1[1],
                        'state': state1[1],
                        'fingertips': fingertips1[1],
                        'ft': ft1[1],
                    }
                    final_paths = {
                        'rgb': rgb2[1],         # at frame_final rgb image
                        'depth': depth2[1],     # at frame_final depth image
                        'state': state2[1],     # at frame_final robot state
                        'fingertips': fingertips2[1],      # ref to final frame finger tips location -> 
                                                            # ref to frame_initial, coordinate of the future finger tips
                        'ft': ft2[1],
                    }

                    # dataset.append(data_point) # the dataset is a list of dictionaries
                    dataset.append((prompt_path, initial_paths, final_paths))

        # shuffling the dataset
        if shuffle:
            np.random.shuffle(dataset)
        else:
            np.array(dataset)
        return dataset

    def get_data_bipartite(self, root_dir: List, shuffle=True):
        """ 
        Get data from the root directory and return a list dictionaries with the file paths for the prompt, initial data, and final data
        Bipartite = load all possible pairs of initial and final frames
        """
        dataset = []
        data_parents = []

        # saving the data parents (one folder per run of capture_data())
        for folder in root_dir:
            data_parent = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    potential_keyframe_path = os.path.join(str(Path(os.path.join(root, file)).parent.parent), 'keyframe_list.npz')
                    potential_ft_calib_path = os.path.join(str(Path(os.path.join(root, file)).parent.parent), 'ft_calibration.npy')
                    if os.path.exists(potential_keyframe_path) and os.path.exists(potential_ft_calib_path): # if the keyframe list and ft calibration file exist
                        data_parent.append(str(Path(os.path.join(root, file)).parent.parent))
            data_parent = list(set(data_parent))
            data_parent.sort(key=lambda x: os.path.getmtime(x))
            if self.args.num_folders > 0 and self.stage == 'train':
                data_parent = data_parent[:self.args.num_folders] # taking only the first num_folders folders if > 0 and training (for data experiments)
            data_parents.append(data_parent)

        data_parents_lengths = [len(data_parent) for data_parent in data_parents]
        print('data_parents_lengths: ', data_parents_lengths)

        # merging the data parents
        data_parents = [item for sublist in data_parents for item in sublist] # list of lists -> list

        # sorting data_parents by order of creation
        data_parents.sort(key=lambda x: os.path.getmtime(x))

        data_parents = list(set(data_parents)) # deduplication

        # print('recording session folders: ', data_parents)
        print('Number of folders: ', len(data_parents))

        # getting the keyframe list for each episode
        for data_folder in data_parents:
            keyframe_index_list = np.load(os.path.join(data_folder, 'keyframe_list.npz'), allow_pickle=True)['keyframe_index_list']
            keyframe_step_list = np.load(os.path.join(data_folder, 'keyframe_list.npz'), allow_pickle=True)['keyframe_step_list']
            prompt_path = os.path.join(data_folder, 'prompt.txt')

            initial_indices = keyframe_index_list[keyframe_step_list == 0]
            final_indices = keyframe_index_list[keyframe_step_list == 1]

            '''
            The dataset contains pairs of (rgb, depth, prompt, state, fingertips, ft), as well as the prompt.
            First, we separate keyframe_index list into pairs of (initial, final) frames based on the keyframe_step_list (elements are either 0 or 1)
            '''
            # for index in range(initial_indices.shape[0]):
            #     # choosing first n initial frames from each folder
            #     # if index < 1:
            # # choosing 5 random initial frames
            # # for index in np.random.choice(initial_indices.shape[0], min(initial_indices.shape[0], 20), replace=False): # TODO: remove this line
            #     for next_index in range(final_indices.shape[0]): # we're saving every possible pair of (initial, final) frames

                # choosing first n initial frames from each folder
                # if index < 1:
            # choosing 5 random initial frames
            # for index in np.random.choice(initial_indices.shape[0], min(initial_indices.shape[0], 20), replace=False): # TODO: remove this line
        
            if self.args.keypoints_per_folder and self.args.keypoints_per_folder < initial_indices.shape[0] and self.stage == 'train':
                num_keypoints = self.args.keypoints_per_folder
            else:
                num_keypoints = initial_indices.shape[0]

            print(f'{num_keypoints} in {data_folder}')

            for next_index in range(final_indices.shape[0]): # we're saving every possible pair of (initial, final) frames
                # for index in range(initial_indices.shape[0]):
                for index in np.random.choice(initial_indices.shape[0], min(initial_indices.shape[0], num_keypoints), replace=False): # taking num_keypoints random initial frames
                    '''
                    initial_paths and final_paths are dictionaries with keys 'rgb', 'depth', 'state', 'fingertips', 'ft'
                    for example, initial_paths['rgb'] is the rgb image of the initial state
                    '''
   
                    initial_paths = {
                        'rgb': os.path.join(data_folder, 'rgb', str(initial_indices[index]) + '.png'),
                        'depth': os.path.join(data_folder, 'depth', str(initial_indices[index]) + '.png'),
                        'state': os.path.join(data_folder, 'state', str(initial_indices[index]) + '.txt'),
                        'fingertips': os.path.join(data_folder, 'fingertips', str(initial_indices[index]) + '.npz'),
                        'ft': os.path.join(data_folder, 'ft', str(initial_indices[index]) + '.npy'),
                    }

                    final_paths = {
                        'rgb': os.path.join(data_folder, 'rgb', str(final_indices[next_index]) + '.png'), # at frame_final rgb image
                        'depth': os.path.join(data_folder, 'depth', str(final_indices[next_index]) + '.png'), # at frame_final depth image
                        'state': os.path.join(data_folder, 'state', str(final_indices[next_index]) + '.txt'), # at frame_final state
                        'fingertips': os.path.join(data_folder, 'fingertips', str(final_indices[next_index]) + '.npz'), # ref to final frame finger tips location -> 
                                                                                                                            # ref to frame_initial, coordinate of the future finger tips
                        'ft': os.path.join(data_folder, 'ft', str(final_indices[next_index]) + '.npy'),
                    }
                    # dataset.append(data_point) # the dataset is a list of dictionaries
                    # checking that fingertips are not the default value TODO: delete
                    # if (np.load(initial_paths['fingertips'], allow_pickle=True)['left'] != -np.ones(3)).all() and \
                    #     (np.load(initial_paths['fingertips'], allow_pickle=True)['right'] != -np.ones(3)).all() and \
                    #     (np.load(final_paths['fingertips'], allow_pickle=True)['left'] != -np.ones(3)).all() and \
                    #     (np.load(final_paths['fingertips'], allow_pickle=True)['right'] != -np.ones(3)).all():

                    # TODO: skip datapoints where the ground truth is outside of the image using cv2.projectPoints

                    data_point = (prompt_path, initial_paths, final_paths)
                    dataset.append(data_point)
                    # else:
                    #     print('DETECTED DEFAULT FINGERTIPS')
                    #     print(np.load(initial_paths['fingertips'], allow_pickle=True)['left'])
                    #     print(np.load(initial_paths['fingertips'], allow_pickle=True)['right'])

        # shuffling the dataset
        if shuffle:
            np.random.shuffle(dataset)
        else:
            dataset = np.array(dataset)
        return dataset

########################################################################

if __name__ == '__main__':
    config, args = parse_config_args()
    shuffle = True
    dataset = ActAffData(args.folder, stage=args.stage, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)

    # pcd_view = PCDViewer(blocking=True)
    
    pcd_view = None
    if args.ros_viz:
        from ros_scripts.ros_viz import RosVizInterface
        pcd_view = RosVizInterface()

    for prompt, initial_data, final_data, rgb_paths in tqdm(loader):
        # print(rgb_paths)
        if args.view:
            visualize_datapoint(prompt, initial_data, final_data,
                                config=config, viz_3d=pcd_view)

        # if (initial_data['left_fingertip'] == -torch.ones((1,3))).all() or \
        #     (initial_data['right_fingertip'] == -torch.ones((1,3))).all() or \
        #     (final_data['left_fingertip'] == -torch.ones((1,3))).all() or \
        #     (final_data['right_fingertip'] == -torch.ones((1,3))).all():

        #     print('DETECTED DEFAULT FINGERTIPS')
        #     visualize_datapoint(prompt, initial_data, final_data,
        #                         config=config, viz_3d=pcd_view)
        #     filter_data(rgb_paths)
        
        print('rgb_paths: ', rgb_paths)
        if args.filter:
            print('rgb_paths', rgb_paths)
            filter_data(rgb_paths)
