#!/usr/bin/env python3

import cv2
from utils.realsense_utils import RealSense, CameraType, PCDViewer

from utils.config_utils import *
from utils.ft_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
from robot.robot_utils import *
from stretch_remote.robot_utils import *
from recording.ft import FTCapture
from stretch_remote.remote_client import RemoteClient
from utils.aruco_detect import ArucoPoseEstimator, find_contact_markers
import sys
import time
import os
import shutil
import json
import re
import datetime

'''
conda deactivate
cd
stretch_robot_home.py 
python3 stretch_remote/stretch_remote/robot_server.py
conda activate cfa
python -m recording.capture_data --config data_collection_5_18 --stage raw --folder medicine_bottle_5_21_frame_1_2 --view 1 --prompt "pick up the medicine bottle" --bipartite 1
python -m prediction.loader --config data_collection_5_18 --stage raw --folder data/raw/medicine_bottle_5_21_frame_1_2 --view 1 --bipartite 1
conda deactivate
cd
python3 stretch_remote/stretch_remote/robot_server.py
'''

HOME_JOINTS = {'y':0.0, 'pitch': 0., 'gripper': 50, 'roll': -0.0, 'yaw': 0.0}

##############################################################################

class DataCapture:
    def __init__(self):
        self.config, self.args = parse_config_args()
        self.data_folder = 'data'

        if not self.check_filename(self.args.folder): # TODO: check this
            print("Please enter a valid folder name ending with frame_x_y_z, when x y and z are ints")
            sys.exit(0)

        self.realsense = RealSense(select_device=self.args.realsense_id, view=self.args.view,
                                   auto_expose=(not self.args.disable_auto_expose))
        self.ft = FTCapture()
        self.fingertips = None

        self.rc = RemoteClient(ip=self.args.ip, home_dict = HOME_JOINTS)
        if self.rc.get_status(compact=True) is None:
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

        self.prompt = self.args.prompt

        # saving prompt to a text file
        with open(os.path.join(self.data_folder, self.args.stage, self.save_folder, 'prompt.txt'), 'w') as f:
            f.write(self.prompt)

        self.data_index = 0
        self.keyframe_index_list = []
        self.keyframe_step = 0
        self.keyframe_step_list = []
        self.keyframe_prompt_list = []
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

        # self.args.folder = self.args.folder + '_' + str(folder_count)
        self.save_folder = self.args.folder + '_' + str(folder_count)

        # setting folder names as class attributes (self.<name>_folder)
        for name in ['rgb', 'depth', 'prompt', 'state', 'fingertips', 'ft']:
            folder = os.path.join(self.data_folder, self.args.stage, self.save_folder, name)
            setattr(self, name + '_folder', folder)
        
        # making directories for data if they doesn't exist
        for folder in [self.rgb_folder, self.depth_folder, self.state_folder, self.fingertips_folder, self.ft_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.folder_count = folder_count

    def save_ft_calibration(self):
        self.ft_offset = self.ft.get_ft()

        if np.abs(self.ft_offset.mean()) < 1e-3:
            print('FT NOT CONNECTED')
            sys.exit()

        print('CALIBRATING FT: ', self.ft_offset)
        time.sleep(0.5)

        np.save(os.path.join(self.data_folder, self.args.stage, self.save_folder, 'ft_calibration.npy'), self.ft_offset)
        print('saving to ', os.path.join(self.data_folder, self.args.stage, self.save_folder, 'ft_calibration.npy'))

    def manage_multiple_prompts(self):
        prompt_list = list(set(self.keyframe_prompt_list)) # deduplication
        print('prompt_list:', prompt_list)
        old_folder = os.path.join('data', self.args.stage, self.save_folder)

        for prompt in prompt_list:
            print('prompt:', prompt)
            # counting the number of folders in the stage folder beginning with the current prompt and args.folder
            folder_indices = [int(f.split('_')[-1]) for f in os.listdir(os.path.join(self.data_folder, self.args.stage)) if f.startswith(f"{prompt.replace(' ', '_')}_{self.args.folder}")]
            print('folder_indices:', folder_indices)
        
            if len(folder_indices) == 0:
                new_folder_index = 0
            else:
              # new folder index is the max of the last number in the folder + 1
              new_folder_index = max(folder_indices) + 1
            new_folder = os.path.join('data', self.args.stage, f"{prompt.replace(' ', '_')}_{self.args.folder}_{new_folder_index}") #TODO: fix this
            print('new folder:', new_folder)
            print('keyframe_index_list', self.keyframe_index_list)
            print('keyframe_step_list:', self.keyframe_step_list)
            print('keyframe_prompt_list:', self.keyframe_prompt_list)

            # copying the contents of old_folder to new_folder
            copy_folder_contents(old_folder, new_folder)
            print('copying contents of ', old_folder, ' to ', new_folder)

            # deleting keyframe_list.npz from new_folder
            os.remove(os.path.join(new_folder, 'keyframe_list.npz'))
            print('deleting ', os.path.join(new_folder, 'keyframe_list.npz'))

            # we now need to get the indices of keyframe_prompt_list that match the current prompt or 
            # if keyframe_step_list is 0 (initial frames applied to all prompts)
            prompt_indices = [i for i, x in enumerate(self.keyframe_prompt_list) if x == prompt or self.keyframe_step_list[i] == 0]
            prompt_indices = np.array(prompt_indices)
            self.keyframe_index_list = np.array(self.keyframe_index_list)
            self.keyframe_step_list = np.array(self.keyframe_step_list)

            new_keyframe_index_list = self.keyframe_index_list[prompt_indices]
            new_keyframe_step_list = self.keyframe_step_list[prompt_indices]
            print('new_keyframe_index_list:', new_keyframe_index_list)
            print('new_keyframe_step_list:', new_keyframe_step_list)
            np.savez(os.path.join(new_folder, 'keyframe_list'), keyframe_index_list=new_keyframe_index_list, keyframe_step_list=new_keyframe_step_list)
            print('saving keyframe_list to ', os.path.join(new_folder, 'keyframe_list.npz'))

            # deleting prompt.txt from new_folder
            os.remove(os.path.join(new_folder, 'prompt.txt'))

            # creating new prompt.txt in new_folder
            with open(os.path.join(new_folder, 'prompt.txt'), 'w') as f:
                f.write(prompt)

        print(f'saved {len(prompt_list)} prompts to <prompt name>_{self.args.folder}')

        # moving old_folder to data/raw
        if self.args.stage != 'raw':
            if not os.path.exists(os.path.join('data', 'raw', 'multiprompt')):
                os.makedirs(os.path.join('data', 'raw', 'multiprompt'))
            # move the old_folder and rename it 
            # shutil.move(old_folder, os.path.join('data', 'raw', 'multiprompt'))
            # get time stamp
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            folder_name = os.path.basename(os.path.normpath(old_folder))
            shutil.copytree(old_folder, 
                            os.path.join('data', 'raw', 'multiprompt', f"{folder_name}-{time_stamp}"))
            shutil.rmtree(old_folder)
            print('backup ', old_folder, ' in ', os.path.join('data', 'raw', 'multiprompt'))

    def capture_data(self, viz_3d=None):
        # get data snapshots
        rgb_image, depth_image = self.realsense.get_rgbd_image()

        if viz_3d is not None:
            pcd = self.realsense.get_point_cloud(rgb_image, depth_image)
            viz_3d.display(pcd)

        ft_data = self.ft.get_ft()

        print('ft: ', ft_data - self.ft_offset)

        detected_fingertips = self.aruco_pose_estimator.get_fingertip_poses(rgb_image)

        # print('left translation:', left_fingertip)
        # print('right translation:', right_fingertip)

        print('self.keyframe:', self.keyframe)

        # need to access robot state to control with keyboard
        keycode = cv2.waitKey(1) & 0xFF
        keyboard_teleop(self.rc, self.config.ACTION_DELTA_DICT, keycode, self)

        image_name = str(self.data_index) + '.png'
        depth_name = image_name
        prompt_name = str(self.data_index) + '.txt'
        state_name = str(self.data_index) + '.txt'
        fingertips_name = str(self.data_index)
        ft_name = str(self.data_index)

        if self.realsense.view:
            disp_time = str(round(self.realsense.current_frame_time - self.realsense.first_frame_time, 3))
            disp_img = self.realsense.display_rgbd_image(rgb_image, depth_image)

            # display time
            cv2.putText(disp_img, disp_time, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 

            # display keyframe list length
            cv2.putText(disp_img, 'keyframe list length: ' + str(len(self.keyframe_index_list)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            
            # display keyframe step
            cv2.putText(disp_img, 'step: ' + str(self.keyframe_step + 1), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

            # display prompt
            cv2.putText(disp_img, 'prompt: ' + str(self.prompt), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)            

            cv2.imshow("frames", disp_img)

        if detected_fingertips is None:
            print('NO FINGERTIPS DETECTED!!! IGNORE This frame \n\n')
            self.keyframe = False

        if self.keyframe:           
            self.keyframe_index_list.append(self.data_index)
            self.keyframe_step_list.append(self.keyframe_step)
            self.keyframe_prompt_list.append(str(self.prompt))

        if (self.keyframe or self.args.save_all_frames) and detected_fingertips is not None:
            # save data to machine
            if self.args.stage in ['train', 'test', 'raw']:
                cv2.imwrite(os.path.join(self.rgb_folder, image_name), rgb_image) # save rgb image
                cv2.imwrite(os.path.join(self.depth_folder, depth_name), depth_image) # save depth image

                robot_status = self.rc.get_status()
                with open(os.path.join(self.state_folder, state_name), 'w') as file: # save robot state
                    file.write(str(robot_status))

                # np.save(os.path.join(self.fingertips_folder, fingertips_name), self.fingertips) # save fingertip poses
                left_fingertip, right_fingertip = detected_fingertips
                np.savez(os.path.join(self.fingertips_folder, fingertips_name), left=left_fingertip, right=right_fingertip)
                np.save(os.path.join(self.ft_folder, ft_name), ft_data) # save ft data

                # save keyframe list
                # np.save(os.path.join(self.data_folder, self.args.stage, self.args.folder, keyframe_name), self.keyframe_index_list)
                np.savez(os.path.join(self.data_folder, self.args.stage, self.save_folder, 'keyframe_list'), keyframe_index_list=self.keyframe_index_list, keyframe_step_list=self.keyframe_step_list)

            else:
                print('Invalid stage argument. Please choose train, test, or raw')
                sys.exit(1)

        if self.delete_last_keyframe:
            self.keyframe_index_list.pop()
            self.keyframe_step_list.pop()
            self.keyframe_prompt_list.pop()
            self.delete_last_keyframe = False

        print('keyframe_index_list: ', self.keyframe_index_list)
        print('keyframe_step: ', self.keyframe_step)
        # print('keyframe_prompt_list:', self.keyframe_prompt_list)
        self.data_index += 1
        print('Average FPS', self.realsense.frame_count / (time.time() - self.realsense.first_frame_time))


##############################################################################


if __name__ == '__main__':
    dc = DataCapture()

    # pcd_vis = PCDViewer()
    while not dc.stop:
        dc.capture_data()

    dc.manage_multiple_prompts()
    folder_sizes = [len(files) for r, d, files in os.walk(os.path.join(dc.data_folder, dc.args.stage, dc.args.folder))][1:]
    folder_names = [r.split('/')[-1] for r, d, files in os.walk(os.path.join(dc.data_folder, dc.args.stage, dc.args.folder))][1:]
    folder_dict = dict(zip(folder_names, folder_sizes))
    
    print('folder sizes: ', folder_dict)

    if len(set(folder_sizes)) > 1:
        print('ERROR: not all folders have the same number of files')
        print('missing files in: ', [k for k, v in folder_dict.items() if v != max(folder_sizes)])


    # depth_example = cv2.imread('data/raw/test_5_8_21/depth/1683597964_278.png', cv2.IMREAD_ANYDEPTH)
    # print(depth_example)
    # print('average depth: ', depth_example.mean())
    # print('min depth: ', depth_example.min())
    # print('max depth: ', depth_example.max())
    # cv2.imshow('depth', depth_example)
    # cv2.waitKey(0)
