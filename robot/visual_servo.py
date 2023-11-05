#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from prediction.models import *
from prediction.loader import ActAffData
from prediction.live_model import LiveModel
from utils.config_utils import *
from utils.pred_utils import *
from utils.data_pipeline import *
from robot.robot_utils import *
from utils.transform import *
from utils.aruco_detect import ArucoPoseEstimator, find_contact_markers
from utils.ft_utils import ft_to_cam_rotation

import time
import numpy as np
import cv2
from utils.realsense_utils import RealSense, CameraType

from stretch_remote.robot_utils import *
from stretch_remote.remote_client import RemoteClient
from typing import Dict, Optional, Tuple

"""
Note: Robot Joint directions (remote client) and coordinates

   ___ z +ve
   ||0
   ||------C                  z +ve ^
   ||                               |  x +ve
   || /x +ve                        | /
  [__] ----> y +ve         y +ve <---/   Robot frame
   / x
"""

#####################################################################################
# CONFIGS

USE_FORCE_OBJECTIVE = True

TEMPORAL_FILTERING_WINDOW = 4
CONTROL_FREQ = 16 # Hz

# HAND_MOVEMENT_ERR_THRESH = 0.023
HAND_MOVEMENT_ERR_THRESH = 0.015
GRIPPER_MOVEMENT_ERR_THRESH = 0.02

WRIST_FORCE_LAMBDA = 0.011
GRIP_FORCE_LAMBDA = 0.013

GRIPPER_EPS = 0.005
GRIPPER_STATIC_OFFSET = -0.005

#####################################################################################

def print_red(text):
    print("\033[91m {}\033[00m".format(text))

def print_green(text):
    print('\033[92m {}\033[00m'.format(text))

#####################################################################################

class StateError:
    trans = np.array([0.0, 0.0, 0.0])
    wrist_force = np.array([0.0, 0.0, 0.0])
    width = 0.0
    grip_force = 0.0
    
    @staticmethod
    def from_np(err):
        """convert from numpy array"""
        assert np.shape(err) == (8,)
        se = StateError()
        se.trans = err[0:3]
        se.wrist_force = err[3:6]
        se.width = err[6]
        se.grip_force = err[7]
        return se

    def to_np(self):
        """convert to numpy array"""
        np_arr = np.zeros(8)
        np_arr[:3] = self.trans
        np_arr[3:6] = self.wrist_force
        np_arr[6] = self.width
        np_arr[7] = self.grip_force
        return np_arr


#####################################################################################

class VisualServo(LiveModel):
    def __init__(self):
        super().__init__()
        self.pos_dict = get_pos_dict(self.rc)
        self.rc.home()
        time.sleep(1)

        # CONFIGS
        self.servo_delta = np.array([0.01, 0.01, 0.01]) # x, y, z
        # self.force_delta = 0.015 # should be similar to above
        # self.grip_servo_factor = 5

        self.servo_min = -0.04
        self.servo_max = 0.04
        self.yaw_limit = [-1.1, 2.8] # abs yaw limit to prevent camera from hitting the wrist
        self.start_grasp_err_norm = 0.075 # 0.06

        self.prev_move_time = time.time()
        self.state_err_history = []

        if self.args.ros_viz:
            from ros_scripts.ros_viz import RosVizInterface
            self.ros_viz = RosVizInterface()
            self.publish_to_rviz = True

    def get_temporal_error(self, err: StateError) -> StateError:
        self.state_err_history.append(err.to_np())
        if len(self.state_err_history) > TEMPORAL_FILTERING_WINDOW:
            self.state_err_history.pop(0)
        err_np = np.array(self.state_err_history)
        err_np = np.mean(err_np, axis=0)
        return StateError.from_np(err_np)

    def proportional_hand_servo_delta(self, distance):
        if distance > 0.2:
            servo_d = self.servo_delta*4
        elif distance > 0.12:
            servo_d = self.servo_delta*2
        else:
            servo_d = self.servo_delta
        return servo_d

    def proportional_gripper_servo_delta(self, distance):
        gripper_servo = 8
        if distance > 0.04:
            servo_d = gripper_servo*3
        elif distance > 0.02:
            servo_d = gripper_servo*2
        else:
            servo_d = gripper_servo
        return servo_d

    def control_robot(self) -> Tuple[Optional[Dict], bool]:
        """
        Return an action when the robot is in the current state, 
        and the error norm
        """
        next_subgoal = False
        # if (time.time() - self.prev_move_time) < 1.0 / CONTROL_FREQ:
        #     return None, next_subgoal

        self.prev_move_time = time.time()

        ############################################################################
        pred_l_contact, pred_r_contact = self.pred_left_fingertip, self.pred_right_fingertip
        curr_l_contact, curr_r_contact = self.curr_left_fingertip, self.curr_right_fingertip
        curr_grip_force = self.grip_force
        pred_grip_force = self.pred_grip_force
        curr_force = self.curr_force[:3] # these forces has been offset-ed
        pred_force = self.pred_force[:3] # these forces has been offset-ed

        if self.pred_left_fingertip is None or self.pred_right_fingertip is None:
            print("Detect Nothing, Do Nothing!")
            return None, next_subgoal

        pred_centroid, pred_width, pred_yaw = fingertips_to_centroid_width_yaw(
            pred_l_contact, pred_r_contact)

        curr_centroid, curr_width, curr_yaw = fingertips_to_centroid_width_yaw(
            curr_l_contact, curr_r_contact)

        print(f"[VS] current pred timestep: {self.pred_timestep_index}")
        print(' [VS]curr_centroid: ', curr_centroid)
        print(' [VS]pred_centroid: ', pred_centroid)
        print(f' [VS]curr vs pred grip width: {curr_width} \t | {pred_width}')
        print(f' [VS]curr vs pred grip force: {curr_grip_force} \t | {pred_grip_force}')
        print(' [VS]pred_force: ', pred_force)
        print(' [VS]curr force: ', curr_force)
        print(' [VS]curr force mag: ', np.linalg.norm(curr_force))
        print('ablate_force: ', self.args.ablate_force)
        print('binary_grip: ', self.args.binary_grip)
        if np.linalg.norm(curr_force) > 30:
            print(' [VS] MAX FORCE REACHED!!!!')

        curr_cam_force = curr_force@ft_to_cam_rotation()
        curr_bot_force = camera_frame_to_robot_frame(self.pos_dict, curr_cam_force)
        pred_cam_force = pred_force@ft_to_cam_rotation()
        pred_bot_force = camera_frame_to_robot_frame(self.pos_dict, pred_cam_force)
        print(' [VS]curr robot frame force: ', curr_bot_force)

        ############################################################################

        trans_err = camera_frame_to_robot_frame(self.pos_dict,
                                               pred_centroid - curr_centroid)
        state_err = StateError()
        state_err.trans = trans_err
        state_err.width = pred_width - curr_width
        state_err.wrist_force = curr_bot_force - pred_bot_force   # opposing force direction to translation
        state_err.grip_force = curr_grip_force - pred_grip_force  # opposing force direction to translation

        avg_state_err = self.get_temporal_error(state_err)
        # print(' [VS]delta robot frame force: ', avg_state_err.trans)
        # print(' [VS]delta robot frame force: ', avg_state_err.wrist_force)

        ############################################################################

        # Move the robot, Kinemtic Objective + Force Objective
        if self.args.ablate_force:
            hand_movement = avg_state_err.trans
        else:
            hand_movement = avg_state_err.trans + WRIST_FORCE_LAMBDA*avg_state_err.wrist_force

        # End Effector Control
        if abs(hand_movement[0]) > HAND_MOVEMENT_ERR_THRESH*3 or abs(hand_movement[2]) > HAND_MOVEMENT_ERR_THRESH*3:
            print("  -------- first line up the robot in x and z-dir -------")
            hand_movement[1] = 0.

        servo_delta = self.proportional_hand_servo_delta(np.linalg.norm(hand_movement))
        hand_movement_delta = servo_delta * hand_movement / np.linalg.norm(hand_movement)
        # print('  @@ hand movement: {hand_movement} | ', np.linalg.norm(hand_movement))
        # print('  @@ movement trans delta: ', hand_movement_delta)

        # set a limit to trans_delta and check if there is NaN # make sure!!
        hand_movement_delta[np.isnan(hand_movement_delta)] = 0
        hand_movement_delta = np.clip(hand_movement_delta, self.servo_min, self.servo_max)

        control_request ={
                    'x': self.pos_dict['x'] + hand_movement_delta[0],
                    'y': self.pos_dict['y'] - hand_movement_delta[1],  # joint space != cartesian space
                    'z': self.pos_dict['z'] + hand_movement_delta[2],
                    'pitch': -0.3
                    # 'pitch': -0.15
                    # 'pitch': -0.0
            }

        ######################################################################
        # Gripper control
        if self.args.ablate_force and not self.args.binary_grip: # continuous grip position with no force
            gripper_movement = avg_state_err.width + GRIPPER_STATIC_OFFSET
        elif self.args.binary_grip: # binary grip, no grip force, could still have wrist force
            print('gripper position: ', self.pos_dict['gripper'])
            WIDTH_THRESH = 0.085
            # CLOSED_POS = -25.0
            CLOSED_POS = -95.0
            # OPEN_POS = 50.0
            OPEN_POS = 50.0

            if pred_width < WIDTH_THRESH and self.pos_dict['gripper'] > CLOSED_POS: # if pred width is small and not closed, close it
                gripper_movement = -1
            elif pred_width > WIDTH_THRESH and self.pos_dict['gripper'] < OPEN_POS: # if pred width is large and not open, open it
                gripper_movement = 1
            else:
                gripper_movement = 0
        else: # continuous grip position with force
            gripper_movement = avg_state_err.width + GRIP_FORCE_LAMBDA*avg_state_err.grip_force + GRIPPER_STATIC_OFFSET
        
        servo_delta = int(self.proportional_gripper_servo_delta(gripper_movement))
        if gripper_movement < -GRIPPER_EPS:
            gripper_control_request = self.pos_dict['gripper'] - servo_delta
        elif gripper_movement > GRIPPER_EPS:
            gripper_control_request = self.pos_dict['gripper'] + servo_delta
        else:
            gripper_control_request = self.pos_dict['gripper'] # do nothing

        if abs(gripper_movement) < GRIPPER_EPS*4:
            control_request['gripper'] = gripper_control_request
        else:
            control_request = {'gripper': gripper_control_request} # only control gripper

        ######################################################################

        if self.args.ros_viz and self.publish_to_rviz:
            """This visualizes the points in 3D space"""
            pcd = get_point_cloud(self.rgb_image, self.depth_image, Intrinsic640())

            # viz_3d.display(pcd, markers=finger_tips)
            self.ros_viz.publish_pcd(pcd, False)
            pred_finger_tips = [np.squeeze(a) for a in [pred_l_contact, pred_r_contact]]
            curr_finger_tips = [np.squeeze(a) for a in [curr_l_contact, curr_r_contact]]

            # self.ros_viz.publish_grip_force(pred_grip_force, centroid)
            # self.ros_viz.publish_fingertips(pred_finger_tips)
            self.ros_viz.publish_wrist_force(pred_cam_force, pred_centroid)
            self.ros_viz.publish_grip_force_with_fingertips(
                pred_grip_force, pred_finger_tips)

            self.ros_viz.publish_wrist_force(curr_cam_force, curr_centroid, is_curr=True)
            self.ros_viz.publish_grip_force_with_fingertips(
                curr_grip_force, curr_finger_tips, is_curr=True)

        ######################################################################
        # Compute error
        # avg_trans_norm = np.linalg.norm(avg_state_err.trans)
        # avg_wrist_force_norm = np.linalg.norm(avg_state_err.wrist_force)

        next_subgoal = True
        hand_movement_norm = np.linalg.norm(hand_movement)
        if abs(hand_movement_norm) < HAND_MOVEMENT_ERR_THRESH:
            print_green(f" @hand movement: {hand_movement} | {hand_movement_norm}")
        else:
            print_red(f" @hand movement: {hand_movement} | {hand_movement_norm}")
            next_subgoal = False

        if abs(gripper_movement) < GRIPPER_MOVEMENT_ERR_THRESH:
            print_green(f" @gripper movement {gripper_movement}")
        else:
            print_red(f" @gripper movement {gripper_movement}")
            next_subgoal = False
            
        if next_subgoal:
            print_green("Switching to next subgoal!")
            self.state_err_history = [] # reset error
            time.sleep(0.5)
        return control_request, next_subgoal

##################################################################################

if __name__ == '__main__':
    vs = VisualServo()
    vs.run_model(control_func=vs.control_robot)
