
import rospy
import tf2_ros as tf
import threading
import json
import numpy as np

from math import pi, atan2
from typing import List, Optional, Tuple

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform, TransformStamped
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Pose, Vector3, Point, Quaternion
import geometry_msgs.msg as gm

from tf import transformations as ts
from visualization_msgs.msg import Marker
import json
import argparse

##############################################################################

def transform_stamped_msg(parent_frame, child_frame, transform):
    t = TransformStamped()
    t.transform = transform
    t.header.frame_id = parent_frame
    t.header.stamp = rospy.Time.now()
    t.child_frame_id = child_frame
    return t

class Pose3D():
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def get_transform(self) -> gm.Transform:
        """convert pose3d to tf2 geometry msg"""
        t = gm.Transform()
        t.translation = gm.Vector3(x=self.x, y=self.y, z=self.z)
        q = ts.quaternion_from_euler(self.roll, self.pitch, self.yaw, 'rxyz')
        t.rotation = gm.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return t
    
    def to_list(self) -> list:
        """convert pose3d to list"""
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

    def to_matrix(self) -> np.ndarray:
        """convert pose3d to transformation matrix"""
        t = self.get_transform()
        rot = t.rotation
        trans = t.translation
        return ts.translation_matrix([trans.x, trans.y, trans.z]) \
               @ ts.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])

##############################################################################

class JointStatePublisher():
    def __init__(self, target_joints):
        rospy.init_node("joint_state_publisher")
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)

        self.br = tf.TransformBroadcaster()
        self.broadcast_tf_lock = threading.Lock() # TODO: impl this
        self.human_pose_transform = None
        self.exercise_start_pose = None
        self.exercise_end_pose = None

        self.joint_states_publisher = rospy.Publisher(
            "/joint_states", JointState, queue_size=1)
        
        assert len(target_joints) == 6, "joints should be a list of 6 floats"
        self.target_joints = target_joints
        self.timer = rospy.Timer(rospy.Duration(1), self._joint_publisher_callback)
        print("joint_state_publisher initialized")

    def _joint_publisher_callback(self, event):
        """
        publish robot joints for visalization test on rviz
        """
        fake_x, lift, arm, wrist_yaw, wrist_pitch, wrist_roll = self.target_joints
        # since the robot is moving on x-axis, we need to adjust the yaw
        # to show the human skeleton correctly
       
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [
            "joint_wrist_pitch", 
            "joint_wrist_yaw",
            "joint_wrist_roll",
            "joint_lift",
            "joint_arm_l3",
            "joint_arm_l2",
            "joint_arm_l1",
            "joint_arm_l0",
        ]
        msg.position = [
            wrist_pitch, wrist_yaw, wrist_roll, lift, arm/4, arm/4, arm/4, arm/4,
        ]
        self.joint_states_publisher.publish(msg)

        tf = Pose3D()
        tf.x = fake_x
        odom_to_baselink = transform_stamped_msg(
            "odom", "base_link", tf.get_transform())
        self.br.sendTransform(odom_to_baselink)


##############################################################################

if __name__ == "__main__":
    # provide example usage in help
    example = "python joint_state_pub.py --joints 0.0 0.0 0.0 0.0 0.0 0.0"
    print("Example usage: ", example)

    parser = argparse.ArgumentParser()
    parser.add_argument("--joints", nargs='+', type=float, help='List of floats joints')
    args = parser.parse_args()

    print("joints: ", args.joints)
    jsp = JointStatePublisher(args.joints)
    rospy.spin()
