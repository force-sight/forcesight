#!/usr/bin/env python3

import time
import sys
import cv2
import numpy as np
import math
import threading

from stretch_remote.remote_client import RemoteClient
from stretch_remote.robot_utils import read_robot_status

from robot.kdl_client import get_inverse_kinematics
import random
from utils.transform import transform_coord, pose_to_mat, mat_to_pose

##############################################################################


def set_angle(angle: float) -> float:
    """
    Ensure the angle is within the range of [-pi, pi] radian convention
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def move_to_target(target_fingertips: np.ndarray,
                   remote_control: RemoteClient,
                   target_pitch=None,
                   move_factor=1.0,  # 0-1.0
                   retries=1  # hack to make sure the robot is in the joint state
                   ) -> bool:
    """
    Provide the target fingertip positions and current joints to move the robot
    to the the target fingertip positions
    This is a non blocking request
    : target_fingertips: np.ndarray of shape (3), reference to cam frame
    : return: True if the action is successful, False otherwise
    """
    s = remote_control.get_status(compact=True)
    current_joints = get_joints_from_robot_status(s)
    current_yaw_joint = current_joints[5]
    # find center of fingertips
    grasp_center = np.mean(target_fingertips, axis=0)
    l_contact, r_contact = target_fingertips[0], target_fingertips[1]
    contact_yaw = math.atan2(r_contact[2] - l_contact[2], r_contact[0] - l_contact[0])
    diff_contact_yaw = contact_yaw + current_yaw_joint

    if diff_contact_yaw < -0.95:
        print("[WARNINIG] YAW is", diff_contact_yaw, "  clip back to -0.95 to make it more stable")
        diff_contact_yaw = -0.95
    elif diff_contact_yaw > 1.58:
        print("[WARNINIG] YAW is", diff_contact_yaw, "  clip back to 1.58 to make it more stable")
        diff_contact_yaw = 1.58

    transformed_coord = transform_coord(grasp_center, current_joints)
    # print(" - grasp center: ", grasp_center)

    # pitch = current_joints[4]
    # yaw = current_joints[3]
    fix_pitch = 0.3
    joints = get_inverse_kinematics(np.append(transformed_coord, [fix_pitch, 0, -1.57 + diff_contact_yaw]))

    # TODO: need to find the exact roll pitch yaw of the target from grasp center
    # r1 = pose_to_mat(np.array([0, 0, 0, 0, 0, -1.57  - 0.8]))
    # r2 = pose_to_mat(np.array([0, 0, 0, 0.2, 0, 0]))
    # pose = mat_to_pose(r1@r2)
    # joints = get_inverse_kinematics(np.append(transformed_coord, pose[3:]))
    if joints is None:
        return False

    if target_pitch is not None:
        # vector from fingertip 0 to fingertip 1
        vec = target_fingertips[1] - target_fingertips[0]

    # NOTE: We are currently not using roll
    target_joints = {}
    target_joints["x"] = (joints[0] - current_joints[0])*move_factor + current_joints[0]
    target_joints["y"] = (joints[2] - current_joints[1])*move_factor + current_joints[1]
    target_joints["z"] = (joints[1] - current_joints[2])*move_factor + current_joints[2] - \
        0.01  # TODO: compensation, might need to fix this
    target_joints["roll"] = 0  # (joints[5] - current_joints[3])*move_factor + current_joints[3]
    target_joints["pitch"] = (joints[4] - current_joints[4])*move_factor + current_joints[4]
    target_joints["yaw"] = (joints[3] - current_joints[5])*move_factor + current_joints[5]
    print(f"[DEBUG TARGET JOINTS]: {joints[0]} {joints[1]} {joints[2]} {joints[3]} {joints[4]} {joints[5]}")

    # calculate the distance between the two fingertips to determine
    # the gripper action
    contact_dist = np.linalg.norm(target_fingertips[0] - target_fingertips[1])
    if contact_dist < 0.04:
        print("[DEBUG] contact_dist is less than 0.04, close gripper")
        target_joints["gripper"] = -20
    elif contact_dist < 0.08:
        target_joints["gripper"] = 5
    else:
        target_joints["gripper"] = 100

    def move_robot_async(target_joints, remote_control, retries):
        print("Run move robot thread...")
        # TODO: hack since the robot is not able to reach the desired joint state
        target_joints_without_grasp = target_joints.copy()
        target_joints_without_grasp.pop("gripper")

        for i in range(retries):
            remote_control.move(target_joints_without_grasp)
            time.sleep(1.3)
        print("run grasp")
        remote_control.move(target_joints)

        s = remote_control.get_status(compact=True)
        joints = get_joints_from_robot_status(s)
        print("----------------------\n After IK action, joints: ", joints)

    thread = threading.Thread(target=move_robot_async, args=(target_joints, remote_control, retries))
    thread.start()
    return True

##############################################################################


def get_pos_dict(rc: RemoteClient):
    """
    Get the current status of the robot from RemoteClient class
    """
    robot_status = rc.get_status(compact=True)
    # print('robot_status: ', robot_status)
    # print('robot_status is not None', robot_status is not None)
    # while True:
    if robot_status is not None:
        pos_dict = robot_status
    else:
        print('cannot read robot status')
        pos_dict = None

    return pos_dict

##############################################################################


def keyboard_teleop(rc, deltas, keycode, self=None):  # enable_moving=True, stop=False):
    if keycode == ord('q') and hasattr(self, 'stop'):     # stop
        self.stop = True
    # if keycode == ord(' ') and hasattr(self, 'enable_moving'):     # toggle moving
    #     self.enable_moving = not self.enable_moving

    if keycode == ord(' ') and hasattr(self, 'keyframe'):  # label as keyframe
        self.keyframe = True
    else:
        self.keyframe = False

    # if backspace is pressed, remove last keyframe
    if keycode == 8 and len(self.keyframe_index_list) > 0:
        self.delete_last_keyframe = True
    else:
        self.delete_last_keyframe = False

    # if enter is pressed, toggle the keyframe step (0 or 1)
    if keycode == 13 and hasattr(self, 'keyframe_step'):
        self.keyframe_step = int(not self.keyframe_step)

    # set the prompt
    if keycode == ord('p') and hasattr(self, 'prompt'):
        self.prompt = input("Enter new prompt: ")
        # self.prompt = self.prompt # NOTE: This might break live model

    move_ok = (self is None or (hasattr(self, 'enable_moving') and self.enable_moving))

    if move_ok:
        if keycode == ord('h'):     # drive home
            rc.home()
        elif keycode == ord(']'):     # drive X
            rc.move({'delta_x': -deltas['x']})
        elif keycode == ord('['):     # drive X
            rc.move({'delta_x': deltas['x']})
        elif keycode == ord('a'):     # drive Y
            pos_dict = get_pos_dict(rc)
            rc.move({'y': pos_dict['y'] - deltas['y']})
        elif keycode == ord('d'):     # drive Y
            pos_dict = get_pos_dict(rc)
            rc.move({'y': pos_dict['y'] + deltas['y']})
        elif keycode == ord('s'):     # drive Z
            pos_dict = get_pos_dict(rc)
            rc.move({'z': pos_dict['z'] - deltas['z']})
        elif keycode == ord('w'):     # drive Z
            pos_dict = get_pos_dict(rc)
            rc.move({'z': pos_dict['z'] + deltas['z']})
        # elif keycode == ord('u'):     # drive roll
        #     rc.move({'roll':pos_dict['roll'] - deltas['roll']})
        # elif keycode == ord('o'):     # drive roll
        #     rc.move({'roll':pos_dict['roll'] + deltas['roll']})
        elif keycode == ord('k'):     # drive pitch
            pos_dict = get_pos_dict(rc)
            rc.move({'pitch': pos_dict['pitch'] - deltas['pitch']})
        elif keycode == ord('i'):     # drive pitch
            pos_dict = get_pos_dict(rc)
            rc.move({'pitch': pos_dict['pitch'] + deltas['pitch']})
        elif keycode == ord('l'):     # drive yaw
            pos_dict = get_pos_dict(rc)
            rc.move({'yaw': pos_dict['yaw'] - deltas['yaw']})
        elif keycode == ord('j'):     # drive yaw
            pos_dict = get_pos_dict(rc)
            rc.move({'yaw': pos_dict['yaw'] + deltas['yaw']})
        elif keycode == ord('b'):     # drive gripper
            pos_dict = get_pos_dict(rc)
            rc.move({'gripper': pos_dict['gripper'] - deltas['gripper']})
        elif keycode == ord('n'):     # drive gripper
            pos_dict = get_pos_dict(rc)
            rc.move({'gripper': pos_dict['gripper'] + deltas['gripper']})
        # elif keycode == ord('1'):     # drive theta
        #     rc.move({'theta':deltas['theta']})
        # elif keycode == ord('2'):     # drive theta
        #     rc.move({'theta':-deltas['theta']})

        if keycode == ord('\\'):
            pos_dict = get_pos_dict(rc)
            rc.move({'z': pos_dict['z'] + deltas['z'] * 10})
        if keycode == ord('='):
            pos_dict = get_pos_dict(rc)
            rc.move({'y': pos_dict['y'] - deltas['y'] * 10})

        # randomize the robot
        if keycode == ord('/'):  # randomize the robot
            arm_random = round(random.uniform(.1, .2), 3)  # .016
            lift_random = round(random.uniform(.8, 1.2), 3)
            pitch_random = round(random.uniform(-.75, .25), 3)
            yaw_random = round(random.uniform(.4, -.4), 3)
            gripper_random = round(random.uniform(-98.0, 60.0), 1)

            print("arm_random: ", arm_random, " lift_random: ", lift_random, " pitch_random: ",
                  pitch_random, " yaw_random: ", yaw_random, "gripper_random: ", gripper_random)
            rc.move({'y': arm_random, 'z': lift_random,
                    'pitch': pitch_random, 'yaw': yaw_random
                     })
            # rc.move({'gripper':gripper_random})
    return keycode


##############################################################################

def level_robot(rc, rpy_eps=0.1, grip_eps=0.5):
    rc.move({'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'gripper': 0})

    print("LEVELING ROBOT")
    time.sleep(1)

    robot_ok, pos_dict = rc.get_status()

    if not robot_ok:
        print("ROBOT NOT CONNECTED")
        return False
    elif abs(pos_dict['roll']) > rpy_eps or abs(pos_dict['pitch']) > rpy_eps or abs(pos_dict['yaw']) > rpy_eps or abs(pos_dict['gripper']) > grip_eps:
        print("ROBOT NOT LEVEL")
        return False
    else:
        print("ROBOT LEVELED")
        return True


def get_joints_from_robot_status(status, joint_sequence=False):
    """
    Utility function to convert rc robot status to joint list
    NOTE: this sequence is not correct, the convertion is in the transform.py
    """
    j = [status['x'], status['z'], status['y'], status['yaw'], status['pitch'], status['roll']]
    print(f"[DEBUG] Proper joint angle seq: {j[0]} {j[1]} {j[2]} {j[3]} {j[4]} {j[5]}")
    if joint_sequence:
        return j
    return [status['x'], status['y'], status['z'],
            status['roll'], status['pitch'], status['yaw']]

##############################################################################


if __name__ == "__main__":
    # TODO: make this configurable
    _d = {
        "x": 0,
        "y": 0.1,
        "z": 0.9067479377560522,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0
    }

    # NOTE: Use your own IP address
    # rc = RemoteClient("100.99.105.59", home_dict = _d)        # RE2
    rc = RemoteClient("100.124.244.50", home_dict=_d)         # RE1
    s = rc.get_status(compact=True)
    joints = get_joints_from_robot_status(s)
    print(" current joints: ", joints)
    # rc.home()

    # time.sleep(3)
    # ## testing target location
    l_fingertip = np.array([0, 0.1, 0.4])
    r_fingertip = np.array([0.05, 0.1, 0.42])

    result = move_to_target([l_fingertip, r_fingertip], rc)
    time.sleep(3)
    print(result)
