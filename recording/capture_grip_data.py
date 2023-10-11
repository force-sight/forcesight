#!/usr/bin/env python3

import cv2
from utils.realsense_utils import RealSense, CameraType, PCDViewer

from utils.config_utils import *
from utils.ft_utils import *
from robot.robot_utils import *
from stretch_remote.robot_utils import *
from recording.ft import FTCapture
from stretch_remote.remote_client import RemoteClient
from utils.aruco_detect import ArucoPoseEstimator, find_contact_markers
import sys
import time
import os
import json
import re

'''
python -m recording.capture_data --config data_collection_5_16 --stage raw --folder test_5_16 --view True --prompt "pick up the mouse" --bipartite True
python -m prediction.loader --config data_collection_5_16 --stage raw --folder data/raw/mouse_5_16_1 --view True --bipartite True
'''

HOME_JOINTS = {'y':0.0, 'pitch': -0.2853, 'gripper': 50, 'roll': -0.0, 'yaw': 0.0}

##############################################################################

class DataCapture:
    def __init__(self):
        self.config, self.args = parse_config_args()
        self.data_folder = 'data'

        if not self.check_filename(self.args.folder): # TODO: check this
            print("Please enter a valid folder name ending with frame_x_y_z, when x y and z are ints")
            sys.exit(0)

        self.realsense = RealSense(select_device=self.args.realsense_id, view=self.args.view)
        self.ft = FTCapture()
        self.fingertips = None

        self.rc = RemoteClient(ip=self.args.ip, home_dict = HOME_JOINTS)
        if self.rc.get_status() is None:
            raise Exception('Remote client not connected')

        self.enable_moving = True
        self.stop = False

        self.manage_folders()

        cam_mat, cam_dist = self.realsense.get_camera_intrinsics(CameraType.COLOR)
        self.aruco_pose_estimator = ArucoPoseEstimator(cam_mat, cam_dist)

        self.rc.move({'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper':0}) # leveling robot
        time.sleep(1)
        self.save_ft_calibration()
        # self.rc.home()
        self.status = self.rc.get_status()

        # saving prompt to a text file
        with open(os.path.join(self.data_folder, self.args.stage, self.args.folder, 'prompt.txt'), 'w') as f:
            f.write(self.args.prompt)

        self.data_index = 0
        self.keyframe_index_list = []
        self.keyframe_step = 0
        self.keyframe_step_list = []
        self.keyframe = False
        self.delete_last_keyframe = False

    def check_filename(self, filename):
        pattern = r"_frame_\d+_\d+$"
        return re.search(pattern, filename)

    def manage_folders(self):
        # making directories for data if they doesn't exist
        for folder in ['data', 'data/raw', 'data/train', 'data/test']:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # counting the number of folders in the stage folder beginning with args.folder
        # folders = os.listdir(os.path.join(self.data_folder, self.args.stage))
        folders = [f for f in os.listdir(os.path.join(self.data_folder, self.args.stage)) if f.startswith(self.args.folder)]
        
        if len(folders) == 0:
            folder_count = 0
        else:
            # folder_count = len([f for f in folders if re.match(self.args.folder, f)])
            # folder_count = len([f for f in folders if f.split('/')[-1].startswith(self.args.folder)])

            # using max of folder names.split('_')[-1] to get the highest folder number
            folder_count = max([int(f.split('_')[-1]) for f in folders if f.split('/')[-1].startswith(self.args.folder)]) + 1
            print('FOLDER COUNT!!!: ', folder_count)

        self.args.folder = self.args.folder + '_' + str(folder_count)

        # setting folder names as class attributes (self.<name>_folder)
        for name in ['rgb', 'depth', 'prompt', 'state', 'fingertips', 'ft']:
            folder = os.path.join(self.data_folder, self.args.stage, self.args.folder, name)
            setattr(self, name + '_folder', folder)
        
        # making directories for data if they doesn't exist
        for folder in [self.rgb_folder, self.depth_folder, self.state_folder, self.fingertips_folder, self.ft_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def save_ft_calibration(self):
        self.ft_offset = self.ft.get_ft()

        if np.abs(self.ft_offset.mean()) < 1e-3:
            print('FT NOT CONNECTED')
            sys.exit()

        print('CALIBRATING FT: ', self.ft_offset)
        time.sleep(0.5)

        np.save(os.path.join(self.data_folder, self.args.stage, self.args.folder, 'ft_calibration.npy'), self.ft_offset)
        print('saving to ', os.path.join(self.data_folder, self.args.stage, self.args.folder, 'ft_calibration.npy'))

    def capture_data(self, viz_3d=None):
        # get data snapshots
        rgb_image, depth_image = self.realsense.get_rgbd_image()

        if viz_3d is not None:
            pcd = self.realsense.get_point_cloud(rgb_image, depth_image)
            viz_3d.display(pcd)

        ft_data = self.ft.get_ft()

        print('ft: ', ft_data - self.ft_offset)

        self.status = self.rc.get_status()

        # # get fingertip poses with aruco, return as np array
        # arucos = self.aruco_pose_estimator.detect(rgb_image) # 11 is left, 12 is right

        # ids = [x.id for x in arucos]

        # if 11 in ids and 12 in ids:
        #     # left_fingertip, right_fingertip = find_contact_markers(arucos[ids.index(11)], arucos[ids.index(12)]) # TODO: move find_contact_markers to __getitem__ in loader           
        #     left_fingertip = np.array(arucos[ids.index(11)].trans)
        #     right_fingertip = np.array(arucos[ids.index(12)].trans)    
        # else:
        #     left_fingertip = -np.ones(3)
        #     right_fingertip = -np.ones(3)

        detected_fingertips = self.aruco_pose_estimator.get_fingertip_poses(rgb_image)

        # print('left translation:', left_fingertip)
        # print('right translation:', right_fingertip)

        # need to access robot state to control with keyboard
        keycode = cv2.waitKey(1) & 0xFF
        keyboard_teleop(self.rc, self.config.ACTION_DELTA_DICT, keycode, self)

        image_name = str(self.data_index) + '.png'
        depth_name = image_name
        prompt_name = str(self.data_index) + '.txt'
        state_name = str(self.data_index) + '.txt'
        fingertips_name = str(self.data_index)
        ft_name = str(self.data_index)
        keyframe_name = 'keyframe_list'        

        if self.realsense.view:
            disp_time = str(round(self.realsense.current_frame_time - self.realsense.first_frame_time, 3))
            disp_img = self.realsense.display_rgbd_image(rgb_image, depth_image)

            # display time
            cv2.putText(disp_img, disp_time, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 

            # display keyframe list length
            cv2.putText(disp_img, 'keyframe list length: ' + str(len(self.keyframe_index_list)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            
            # display keyframe step
            cv2.putText(disp_img, 'step: ' + str(self.keyframe_step + 1), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            cv2.imshow("frames", disp_img)

        if detected_fingertips is None:
            print('NO FINGERTIPS DETECTED!!! IGNORE This frame \n\n')
            self.keyframe = False

        if self.keyframe:
            self.keyframe_index_list.append(self.data_index)
            self.keyframe_step_list.append(self.keyframe_step)

        if (self.keyframe or self.args.save_all_frames) and detected_fingertips is not None:
            # save data to machine
            if self.args.stage in ['train', 'test', 'raw']:
                cv2.imwrite(os.path.join(self.rgb_folder, image_name), rgb_image) # save rgb image
                cv2.imwrite(os.path.join(self.depth_folder, depth_name), depth_image) # save depth image

                with open(os.path.join(self.state_folder, state_name), 'w') as file: # save robot state
                    file.write(str(self.status))

                print('pos_dict: ', get_pos_dict(self.rc))
                left_fingertip, right_fingertip = detected_fingertips

                # np.save(os.path.join(self.fingertips_folder, fingertips_name), self.fingertips) # save fingertip poses
                np.savez(os.path.join(self.fingertips_folder, fingertips_name), left=left_fingertip, right=right_fingertip)
                np.save(os.path.join(self.ft_folder, ft_name), ft_data) # save ft data

                # save keyframe list
                # np.save(os.path.join(self.data_folder, self.args.stage, self.args.folder, keyframe_name), self.keyframe_index_list)
                np.savez(os.path.join(self.data_folder, self.args.stage, self.args.folder, keyframe_name), keyframe_index_list=self.keyframe_index_list, keyframe_step_list=self.keyframe_step_list)

            else:
                print('Invalid stage argument. Please choose train, test, or raw')
                sys.exit(1)

        if self.delete_last_keyframe:
            self.keyframe_index_list.pop()
            self.keyframe_step_list.pop()
            self.delete_last_keyframe = False

        print('keyframe_index_list: ', self.keyframe_index_list)
        print('keyframe_step: ', self.keyframe_step)
        
        self.data_index += 1

        result = {
                  'rgb':rgb_image,
                  'depth':depth_image,
                  'prompt':self.args.prompt,
                  'state':self.status,
                  'fingertips':self.fingertips,
                  'ft_frame':ft_data,
                  'ft_frame_time':self.ft.current_frame_time,
                  'frame_time':self.realsense.current_frame_time,
                  }

        print('Average FPS', self.realsense.frame_count / (time.time() - self.realsense.first_frame_time))

        return result


##############################################################################

if __name__ == '__main__':
    dc = DataCapture()
    delay = []

    # pcd_vis = PCDViewer()
    while not dc.stop:
        data = dc.capture_data()
        delay.append(data['ft_frame_time'] - data['frame_time'])

    folder_sizes = [len(files) for r, d, files in os.walk(os.path.join(dc.data_folder, dc.args.stage, dc.args.folder))][1:]
    folder_names = [r.split('/')[-1] for r, d, files in os.walk(os.path.join(dc.data_folder, dc.args.stage, dc.args.folder))][1:]
    folder_dict = dict(zip(folder_names, folder_sizes))
    
    print('folder sizes: ', folder_dict)

    if len(set(folder_sizes)) > 1:
        print('ERROR: not all folders have the same number of files')
        print('missing files in: ', [k for k, v in folder_dict.items() if v != max(folder_sizes)])
        
    print('saved results to {}'.format(os.path.join(dc.data_folder, dc.args.stage, dc.args.folder)))
    print("delay avg:", np.mean(delay))
    print("delay std:", np.std(delay))
    print("delay max:", np.max(delay))

    # depth_example = cv2.imread('data/raw/test_5_8_21/depth/1683597964_278.png', cv2.IMREAD_ANYDEPTH)
    # print(depth_example)
    # print('average depth: ', depth_example.mean())
    # print('min depth: ', depth_example.min())
    # print('max depth: ', depth_example.max())
    # cv2.imshow('depth', depth_example)
    # cv2.waitKey(0)
