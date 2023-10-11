#!/usr/bin/env python3

from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import List, Optional, Tuple

# NOTE: migrated from kdl to utils/kinpy_wrapper.py
from robot.kinpy_wrapper import get_forward_kinematics

########################################################################

def pose_to_mat(pose: List[float]) -> np.ndarray:
    """Convert a xyzrpy list to a 4x4 matrix"""
    assert len(pose) == 6
    r = R.from_euler('xyz', [pose[3], pose[4], pose[5]], degrees=False)
    rot = r.as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[0, 3] = pose[0]
    mat[1, 3] = pose[1]
    mat[2, 3] = pose[2]
    return mat

def inverse_mat(mat):
    return np.linalg.inv(mat)

def matmul_mat(mat1, mat2):
    return np.matmul(mat1, mat2)

def mat_to_pose(mat):
    "4x4 matrix to xyzrpy list, not using tf"
    translation = mat[:3, 3]
    rotation = R.from_matrix(mat[:3, :3])
    return np.concatenate((translation, rotation.as_euler('xyz', degrees=False)))

def tf_from_mats(mat1, mat2):
    """
    Compute the transform from mat1 to mat2
    """
    return matmul_mat(inverse_mat(mat1), mat2)

def stretch_joint_sequence(joints):
    # convert robot xyzrpy to stretch custom joints sequence
    # NOTE: this is a hack, need a better function design
    return [joints[0], joints[2], joints[1], joints[5], joints[4], joints[3]]

########################################################################

def calc_tf_delta(initial_joint, final_joint):
    """
    Calculate the transform from initial to final joint
    """
    # this requires the kdl server to be running
    init_cam_pose = get_forward_kinematics(
        stretch_joint_sequence(final_joint))
    final_cam_pose = get_forward_kinematics(
        stretch_joint_sequence(initial_joint))
    init_cam_pose = pose_to_mat(init_cam_pose)
    final_cam_pose = pose_to_mat(final_cam_pose)
    return tf_from_mats(final_cam_pose, init_cam_pose)

def get_transformed_fingertips(
        init_state, final_state, final_left_fingertip, final_right_fingertip,
        optional_camera_tf: Optional[np.array] = None
    ) -> Tuple[np.array, np.array]:
    """
    Generate the transformed fingertip positions in the initial frame
    NOTE: this requires the kdl server to be running
    NOTE: the state is in the order of robot's [x, y, z, roll, pitch, yaw]
    :arg optional_camera_tf: the transform of the new camera after custom
                             augmenting the image (e.g. translation and cropping)
    :return fl_coor, fr_coor: fingertip positions in the initial frame
    """
    _f_s = final_state.cpu().numpy()
    _i_s = init_state.cpu().numpy()
   
    # _delta_tf = calc_tf_delta(_f_s, _i_s)
    init_cam_pose = get_forward_kinematics(
        stretch_joint_sequence(_i_s))
    final_cam_pose = get_forward_kinematics(
        stretch_joint_sequence(_f_s))
    init_cam_pose = pose_to_mat(init_cam_pose)

    if optional_camera_tf is not None:
        init_cam_pose = init_cam_pose@optional_camera_tf

    final_cam_pose = pose_to_mat(final_cam_pose)
    _delta_tf =  tf_from_mats(final_cam_pose, init_cam_pose)

    # np add 3 more 0s to the end of the pose
    final_left_fingertip = np.append(final_left_fingertip, [0, 0, 0])
    fl_mat = pose_to_mat(final_left_fingertip)
    final_right_fingertip = np.append(final_right_fingertip, [0, 0, 0])
    fr_mat = pose_to_mat(final_right_fingertip)
    
    fl_diff_mat = tf_from_mats(_delta_tf, fl_mat)
    fr_diff_mat = tf_from_mats(_delta_tf, fr_mat)
    fl_coor = np.array(mat_to_pose(fl_diff_mat)[:3])
    fr_coor = np.array(mat_to_pose(fr_diff_mat)[:3])
    return fl_coor, fr_coor

def transform_coord(coord: np.array, curr_joints, from_cam_to_world=True):
    """
    This function is used to transform the target (e.g. fingertip)
    coordinates from the camera frame to the world frame or vice versa
    :arg coord: target coordinates [x, y, z] in cam frame if True
    :arg curr_joints: current joint angles [x, lift, arm, yaw, pitch, roll]
    """
    target_pose = np.append(coord, [0, 0, 0])
    target_pose = pose_to_mat(target_pose)

    # print(" - target_pose", target_pose)
    cam_pose = get_forward_kinematics(stretch_joint_sequence(curr_joints))
    print(" - cam_pose", cam_pose)
    cam_pose = pose_to_mat(cam_pose)

    if from_cam_to_world:
        new_pose = cam_pose@target_pose
    else:
        new_pose = inverse_mat(target_pose)@cam_pose
    coord = np.array(mat_to_pose(new_pose)[:3])
    print(" - target coord", coord)
    return coord

def camera_frame_to_robot_frame(pos_dict, vec):
    # rotating -100 around robot x axis, 10 degree is camera angle
    rad_100 = -100 * np.pi / 180
    rad_100 += pos_dict['pitch']
    rot_x_100 = np.array([[1, 0, 0],
                        [0, np.cos(rad_100), -np.sin(rad_100)],
                        [0, np.sin(rad_100), np.cos(rad_100)]]) 
    
    # rotating 180 around robot z axis
    rot_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                        [np.sin(np.pi), np.cos(np.pi), 0],
                        [0, 0, 1]])

    return rot_z_180 @ rot_x_100 @ vec

def robot_frame_to_camera_frame(pos_dict, vec):
    # rotating 180 around robot z axis
    rot_z_180 = np.array([[np.cos(np.pi), -np.sin(np.pi), 0],
                        [np.sin(np.pi), np.cos(np.pi), 0],
                        [0, 0, 1]])

    # rotating 100 around robot x axis,  10 degree is camera angle
    rad_100 = 100 * np.pi / 180
    rad_100 += pos_dict['pitch']
    rot_x_100 = np.array([[1, 0, 0],
                        [0, np.cos(rad_100), -np.sin(rad_100)],
                        [0, np.sin(rad_100), np.cos(rad_100)]]) 
    return rot_x_100 @ rot_z_180 @ vec
