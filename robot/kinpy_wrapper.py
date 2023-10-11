#!/usr/bin/env python3

import kinpy as kp
from kinpy.frame import Frame, Link, Joint
from kinpy.chain import Chain, SerialChain
from kinpy.transform import Transform
import transformations as tf


class KinpyWrapper:
    def __init__(self, urdf_file) -> None:

        chain = kp.build_chain_from_urdf(open(urdf_file).read())

        fake_x_frame = Frame(
            name="link_fake_x_frame",
            link=Link("link_fake_x"),
            joint=Joint("joint_fake_x", axis=[1, 0, 0], joint_type="prismatic"),
            children=[chain._root]
        )

        root_base = Frame(
            name="link_new_base_frame",
            link=Link("link_new_base"),
            joint=Joint("joint_new_base", joint_type="fixed", offset=Transform([0, 0, 0])),
            children=[fake_x_frame]
        )

        new_chain = Chain(root_base)
        self.sc = SerialChain(new_chain,
                              end_frame_name="egocam_link_frame",
                              root_frame_name="link_new_base_frame"
                              )

    def forward_kinematics(self, joints) -> list:
        """
        compute forward kinematics
        :param joints: list of joint values [j0, j1, ... jN]
        :return: [x, y, z, roll, pitch, yaw]
        """
        joint_names = self.sc.get_joint_parameter_names()
        assert len(joints) == len(joint_names)
        joint_dict = {}
        for i in range(len(joints)):
            joint_dict[joint_names[i]] = joints[i]
        solution = self.sc.forward_kinematics(joint_dict)
        euler = tf.euler_from_quaternion(solution.rot)
        return list(solution.pos) + list(euler)

#################################################################################
"""
TODO: this is a bad implementation for init in a global space
the reason of doing this is maintain api consistency with kdl_client.py
"""
__urdf_file = "robot/stretch_robot.urdf"
__kinpy_wrapper = KinpyWrapper(__urdf_file)

def get_forward_kinematics(joint_6dofs):
    """
    Simple wrapped function to get forward kinematics
    """
    assert len(joint_6dofs) == 6
    arm_seg = joint_6dofs[2]/4.0
    joints = [
            joint_6dofs[0], joint_6dofs[1],
            arm_seg, arm_seg, arm_seg, arm_seg,
            joint_6dofs[3], joint_6dofs[4], joint_6dofs[5],
        ]
    return __kinpy_wrapper.forward_kinematics(joints)

#################################################################################

def __test_ik():
    """
    Simple test to compare the forward kinematics of KDL and Kinpy
    """
    from robot.kdl_wrapper import KdlWrapper, KdlType
    import numpy as np

    joints = [0.2]*9


    kdl = KdlWrapper.make(KdlType.FULL_ROBOT_6DOF, __urdf_file, "egocam_link")
    origin_pose = kdl.forward_kinematics(joints)

    pose = __kinpy_wrapper.forward_kinematics(joints)
    print(" ----------------------------------------------------")
    print(origin_pose)
    print(pose)
    assert np.allclose(origin_pose, pose), "Incorrect forward kinematics!"


if __name__ == "__main__":
    __test_ik()
