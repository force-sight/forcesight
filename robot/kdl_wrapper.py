#!/usr/bin/env python3

from __future__ import annotations

import kdl_parser_py.urdf
import PyKDL as pykdl
from math import pi as PI
from typing import List, Optional
from tf import transformations as ts
from scipy.spatial.transform import Rotation as R
from utils.transform import mat_to_pose, pose_to_mat
import numpy as np

########################################################################
def list_to_jnt_array(l: List):
    """Convert a list to a KDL JntArray"""
    jnt_array = pykdl.JntArray(len(l))
    for i in range(len(l)):
        jnt_array[i] = l[i]
    return jnt_array

########################################################################
class KdlType:
    WRIST_3DOF = 0
    WRIST_6DOF = 1
    FULL_ROBOT_5DOF = 2
    FULL_ROBOT_6DOF = 3  # this has a fake x joint with total 9 joints

########################################################################
class KdlWrapper:
    @staticmethod
    def make(
            type: KdlType, urdf_file: str, target_frame: str
        ) -> Optional[KdlWrapper]:
        """
        Create a KdlWrapper object for the given sws type.
        @return KdlWrapper object or None if failed.
        """
        kdl_wrapper = KdlWrapper()
        if kdl_wrapper.__create_chain__(type, urdf_file, target_frame):
            print("< Done init KDLWrapper >")
            return kdl_wrapper
        return None

    def number_of_joints(self) -> int:
        """
        Return the number of joints in the chain.
        """
        return self.total_joints

    def forward_kinematics(
            self, joint_values: List[float]
        )-> Optional[List]:
        """
        Compute the forward kinematics for the given joint values.
        @joint_values: [j0, j1, ... jN]
        @return: [x, y, z, roll, pitch, yaw]
        """
        if len(joint_values) != self.total_joints:
            print(f"Error! Expect {self.total_joints} of joints")
            return None
        joints = list_to_jnt_array(joint_values)
        frame_out = pykdl.Frame()
        result = self.fk_solver.JntToCart(joints, frame_out, -1)
        if result < 0:
            print(f"Error! Failed to compute FK [{result}]")
            return None

        pos = frame_out.p
        rot = frame_out.M.GetRPY()
        return [pos[0], pos[1], pos[2], rot[0], rot[1], rot[2]]

    def inverse_kinematics(
            self,
            target_position: List[float],
            combine_arm_extension: bool = False
        ) -> Optional[List]:
        """
        Compute the inverse kinematics for the given target position.
        @target_position: [x, y, z, roll, pitch, yaw]
        @combine_arm_extension: If True, all 4 arm translation joints 
                            will be combined into a single y trans
        @return: [j0, j1, ... jN], or 
                 [fake_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll]
        """
        if len(target_position) != 6:
            print("Error! Expect 6 values for target position")
            return None

        x, y, z, roll, pitch, yaw = target_position
        q = ts.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
        rot = ts.quaternion_matrix(q)
        pos_kdl = pykdl.Vector(x, y, z)
        rot_kdl = pykdl.Rotation(rot[0, 0], rot[0, 1], rot[0, 2],
                                rot[1, 0], rot[1, 1], rot[1, 2],
                                rot[2, 0], rot[2, 1], rot[2, 2])
        frame_kdl = pykdl.Frame(rot_kdl, pos_kdl)

        # Initial guess
        if self.type == KdlType.WRIST_6DOF:
            q_kdl = list_to_jnt_array([x, y, z, roll, pitch, yaw])
        else:
            q_kdl = pykdl.JntArray(self.total_joints)

        q_kdl_out = pykdl.JntArray(self.total_joints)
        result = self.ik_solver.CartToJnt(q_kdl, frame_kdl, q_kdl_out)

        if result < 0:
            print(f"Error! Failed to compute IK [{result}]")
            return None

        joints = [q_kdl_out[i] for i in range(self.total_joints)]

        if combine_arm_extension:
            print("WARNING! Combining arm translation joints!")
            # Combine the 4 arm translation joints into a single y-trans
            if self.type == KdlType.FULL_ROBOT_5DOF:
                # return 5dof (z, y, wrist: yaw, pitch, roll)
                return joints[:1] + [sum(joints[1:5])] + joints[5:]
            elif self.type == KdlType.FULL_ROBOT_6DOF:
                # return 6dof (x, z, y, wrist: yaw, pitch, roll)
                return joints[:2] + [sum(joints[2:6])] + joints[6:]

        return joints

    ########################################################################
    # Private methods
    ########################################################################

    def __init__(self):
        pass

    def __create_chain__(self, type: KdlType, urdf_file: str, target_frame: str):
        self.type = type

        (ok, tree) = kdl_parser_py.urdf.treeFromFile(urdf_file)
        if not ok:
            print("Failed to parse urdf file")
            return False
        
        if type == KdlType.WRIST_3DOF or type == KdlType.WRIST_6DOF:
            parent_frame = "link_arm_l0"
        elif type == KdlType.FULL_ROBOT_5DOF or type == KdlType.FULL_ROBOT_6DOF:
            parent_frame = "base_link"

        ee_chain = tree.getChain(parent_frame, target_frame)
        # Add new fake joints to the end effector chain
        self.new_ee_chain = pykdl.Chain()

        # Add x-y-z translation joints
        if type == KdlType.WRIST_6DOF:
            self.new_ee_chain.addSegment(
                pykdl.Segment("link_fake_x",
                            pykdl.Joint("joint_fake_x", pykdl.Joint.TransX),
                            pykdl.Frame(pykdl.Vector(0, 0, 0)))
            )
            self.new_ee_chain.addSegment(
                pykdl.Segment("link_fake_y",
                            pykdl.Joint("joint_fake_y", pykdl.Joint.TransY),
                            pykdl.Frame(pykdl.Vector(0, 0, 0)))
            )
            self.new_ee_chain.addSegment(
                pykdl.Segment("link_fake_z",
                            pykdl.Joint("joint_fake_z", pykdl.Joint.TransZ),
                            pykdl.Frame(pykdl.Vector(0, 0, 0)))
            )
        # Add x translation joint as the base
        elif type == KdlType.FULL_ROBOT_6DOF:
            self.new_ee_chain.addSegment(
                pykdl.Segment("link_fake_x",
                            pykdl.Joint("joint_fake_x", pykdl.Joint.TransX),
                            pykdl.Frame(pykdl.Vector(0, 0, 0)))
            )
        self.new_ee_chain.addChain(ee_chain)
        print("New chain:")
        print(self.new_ee_chain.getNrOfSegments())
        self.total_joints = self.new_ee_chain.getNrOfJoints()

        if type == KdlType.WRIST_3DOF:
            assert self.total_joints == 3, "Error in chain creation"
        elif type == KdlType.WRIST_6DOF:
            assert self.total_joints == 6, "Error in chain creation"
        elif type == KdlType.FULL_ROBOT_5DOF:
            assert self.total_joints == 8, "Error in chain creation"
        elif type == KdlType.FULL_ROBOT_6DOF:
            assert self.total_joints == 9, "Error in chain creation"

        self.fk_solver = pykdl.ChainFkSolverPos_recursive(self.new_ee_chain)

        # create joint limits
        if type == KdlType.WRIST_3DOF:
            min_joints_list = [-PI*2]*3
            max_joints_list = [PI*2]*3
        elif type == KdlType.WRIST_6DOF:
            min_joints_list = [-2.0]*3 + [-PI*2]*3
            max_joints_list = [2.0]*3 + [PI*2]*3
        elif type == KdlType.FULL_ROBOT_5DOF:
            min_joints_list = [-2.0]*5 + [-PI*2]*3
            max_joints_list = [2.0]*5 + [PI*2]*3
        elif type == KdlType.FULL_ROBOT_6DOF:
            min_joints_list = [-2.0]*6 + [-PI*2]*3
            max_joints_list = [2.0]*6 + [PI*2]*3
            # min_joints_list[1] = 0.22 # lift min limit of the robot

        assert len(min_joints_list) == self.total_joints, "Error create joint limits"
        assert len(max_joints_list) == self.total_joints, "Error create joint limits"

        min_joints = list_to_jnt_array(min_joints_list)
        max_joints = list_to_jnt_array(max_joints_list)

        self.ik_v_kdl = pykdl.ChainIkSolverVel_pinv(self.new_ee_chain)
        self.ik_solver = pykdl.ChainIkSolverPos_NR_JL(
            self.new_ee_chain, min_joints, max_joints,
            self.fk_solver, self.ik_v_kdl, 200, 1e-5)

        return True


##############################################################################

if __name__ == "__main__":
    kdl = KdlWrapper.make(
            KdlType.FULL_ROBOT_6DOF,
            "stretch_robot.urdf",
            "egocam_link",
        )
    
    # Joints representation (1x9):  
    # [ 
    #   fake-x, z-lift,
    #   y-arm1, y-arm2, y-arm3, y-arm4, 
    #   wrist_yaw, wrist_pitch, wrist_roll
    # ]
    joints = [0.3]*9
    origin_pose = kdl.forward_kinematics(joints)
    print(origin_pose)
    mat = pose_to_mat(origin_pose)
    pose = mat_to_pose(mat)
    print("scipy: \n", pose)
