#!/usr/bin/env python3

import argparse
from scipy.spatial.transform import Rotation as R
import math
import numpy as np
from ctypes import * # convert float to uint32

import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Vector3
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs.point_cloud2 as pc2

from recording.ft import FTCapture
from utils.ft_utils import *
from utils.transform import *
from utils.realsense_utils import *

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8

##############################################################################

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="map", invert_color=False):
    """
    Refer to here:
    https://github.com/felixchenfy/open3d_ros_pointcloud_conversion/issues/6

    # TODO: enable down sampling
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asarray(open3d_cloud.points)
    if not open3d_cloud.colors: # XYZ only
        fields=FIELDS_XYZ
        cloud_data=points
    else: # XYZ + RGB
        fields=FIELDS_XYZRGB
        # -- Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors)*255)
        colors = colors.astype(np.uint32)
        if invert_color:
            colors = colors[:,0] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,2]
        else:
            colors = colors[:,2] * BIT_MOVE_16 +colors[:,1] * BIT_MOVE_8 + colors[:,1]
        colors = colors.view(np.float32)
        cloud_data = np.column_stack((points, colors))
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

##############################################################################

class RosVizInterface:
    def __init__(self):
        # Create a ROS node
        # Run rosviz, make sure roscore is running
        rospy.init_node('force_sight_viz_publisher', anonymous=True)
        self.cloud_pub = rospy.Publisher('point_cloud_topic', PointCloud2, queue_size=10)
        self.fingertips_pub = rospy.Publisher("/contact_marker", Marker, queue_size=10)
        self.grip_force_pub = rospy.Publisher("/grip_force_marker", Marker, queue_size=10)

        self.wrist_force_pub = rospy.Publisher("/wrist_force_marker", Marker, queue_size=10)
        self.grip_force_fingers_pub = rospy.Publisher(
            "/grip_force_fingers_markers", MarkerArray, queue_size=10)
        
        self.curr_wrist_force_pub = rospy.Publisher("/curr_wrist_force_marker", Marker, queue_size=10)
        self.curr_grip_force_fingers_pub = rospy.Publisher(
            "/curr_grip_force_fingers_markers", MarkerArray, queue_size=10)

        self.ref_frame = "camera"

        static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = 'map'
        transform.child_frame_id = self.ref_frame

        # Set translation (assuming no translation)
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.1

        # Set rotation
        x_angle = math.pi/2
        y_angle = math.pi
        z_angle = 0
        
        # from euler to rotation matrix
        rot1 = R.from_euler('xyz', [x_angle, y_angle, z_angle]) # arm
        rot2 = R.from_euler('xyz', [-math.pi/6, 0, 0]) # camera
        quaternion = (rot1*rot2).as_quat()

        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]

        # Publish the transform
        static_broadcaster.sendTransform(transform)
        
        # transform.header.frame_id = 'arm'
        # transform.child_frame_id = self.ref_frame
        # transform.transform.translation.z = 0.02

        # quaternion = R.from_euler('xyz', [math.pi/5, 0, 0]).as_quat()
        # transform.transform.rotation.x = quaternion[0]
        # transform.transform.rotation.y = quaternion[1]
        # transform.transform.rotation.z = quaternion[2]
        # transform.transform.rotation.w = quaternion[3]

        # # Publish the transform
        # static_broadcaster.sendTransform(transform)

    def publish_pcd(self, pcd, invert_color=False):
        cloud_msg = convertCloudFromOpen3dToRos(pcd, self.ref_frame, invert_color)
        self.cloud_pub.publish(cloud_msg)
    
    def publish_fingertips(self, coors, size=0.005):
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.action = Marker.ADD  # set the marker action to ADD
        marker.color.a = 0.8  # set the alpha
        marker.color.r = 1.0
        marker.color.b = 1.0

        marker.type = Marker.POINTS  # set the marker type to POINTS
        marker.scale.x = size*2
        marker.scale.y = size*2
        marker.scale.z = size*2

        for c in coors:
            point = Point()
            point.x = c[0]
            point.y = c[1]
            point.z = c[2]
            marker.points.append(point)
        self.fingertips_pub.publish(marker)

    def publish_wrist_force(self, force_vec, origin, scale=0.05, is_curr=False):
        # Invert the force vector
        force_vec = -force_vec

        # Convert force_vec to a unit vector
        force_unit_vec = force_vec / np.linalg.norm(force_vec)

        # Default arrow direction is along x-axis
        arrow_dir = np.array([1, 0, 0])
        # Compute rotation axis (unit vector)
        rotation_axis = np.cross(arrow_dir, force_unit_vec)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Compute rotation angle
        rotation_angle = np.arccos(np.dot(arrow_dir, force_unit_vec))

        # Compute quaternion from axis-angle
        rotation_quat = np.zeros(4)
        rotation_quat[0] = rotation_axis[0] * np.sin(rotation_angle / 2)
        rotation_quat[1] = rotation_axis[1] * np.sin(rotation_angle / 2)
        rotation_quat[2] = rotation_axis[2] * np.sin(rotation_angle / 2)
        rotation_quat[3] = np.cos(rotation_angle / 2)

        # Compute arrow's position at the head
        arrow_length = np.linalg.norm(force_vec)*scale
        arrow_pos = origin - arrow_length*force_unit_vec

        # Create a Marker message
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.type = Marker.ARROW
        marker.pose.position.x = arrow_pos[0]
        marker.pose.position.y = arrow_pos[1]
        marker.pose.position.z = arrow_pos[2]
        marker.pose.orientation.x = rotation_quat[0]
        marker.pose.orientation.y = rotation_quat[1]
        marker.pose.orientation.z = rotation_quat[2]
        marker.pose.orientation.w = rotation_quat[3]
        marker.scale.x = arrow_length  # Length of the arrow
        marker.scale.y = 0.2*scale  # Width of the arrow
        marker.scale.z = 0.2*scale  # Height of the arrow

        if is_curr:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # Yellow
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Yellow

        if is_curr:
            self.curr_wrist_force_pub.publish(marker)
        else:
            self.wrist_force_pub.publish(marker)
            
    def publish_grip_force(self, force_magnitude, origin, force_scale=0.2, sphere_scale=.04):
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD  # set the marker action to ADD
        marker.pose.position.x = origin[0]
        marker.pose.position.y = origin[1]
        marker.pose.position.z = origin[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = sphere_scale
        marker.scale.y = sphere_scale
        marker.scale.z = sphere_scale

        # Calculate the color based on magnitude
        color = ColorRGBA()
        color.r = min(1.0, force_magnitude * force_scale)  # Green to red transition
        color.g = 1.0 - color.r
        color.b = 0.0
        color.a = 0.8
        marker.color = color
        self.grip_force_pub.publish(marker)
        
    def publish_grip_force_with_fingertips(self,
                                           force_magnitude,
                                           fingertips,
                                           force_scale=0.01,
                                           is_curr=False):
        """
        Publishes the grip force as a arrow marker with the fingertips as sphere markers
        """
        # create 2 arrow markers from the fingertips (as origin)
        # to the center, which the arrow is horizontal placed on the x-axis
        # and the length is the force magnitude
        markers_msg = MarkerArray()
        for i in range(2):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = self.ref_frame
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = force_magnitude*force_scale
            marker.scale.y = 0.01  # Adjust as needed for the arrow's width
            marker.scale.z = 0.015 # Adjust as needed for the arrow's height
            marker.pose
            
            if is_curr:
                # marker.color = ColorRGBA(0, 0.6, 1.0, 0.8)  # Light blue opaque
                marker.color = ColorRGBA(0.4, 1.0, 0.1, 0.5)  # Light green
            else:
                # marker.color = ColorRGBA(1.0, 0.2, 1.0, 0.8)
                marker.color = ColorRGBA(0.4, 1.0, 0.1, 1.0)  # Light green
            if i == 0:
                marker.pose.position.x = fingertips[i][0] - force_magnitude*force_scale
            else:
                marker.pose.position.x = fingertips[i][0] + force_magnitude*force_scale
            marker.pose.position.y = fingertips[i][1]
            marker.pose.position.z = fingertips[i][2]
            # marker.pose.orientation.w = 1.0
            
            # flip the arrow direction
            if i == 0:
                marker.pose.orientation.w = 1.0
            else:
                quat = R.from_euler('xyz', [0, 0, np.pi]).as_quat()
                marker.pose.orientation.x = quat[0]
                marker.pose.orientation.y = quat[1]
                marker.pose.orientation.z = quat[2]
                marker.pose.orientation.w = quat[3]
            markers_msg.markers.append(marker)

        for i in range(2):
            marker = Marker()
            marker.id = i + 2
            marker.header.frame_id = self.ref_frame
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD  # set the marker action to ADD
            marker.pose.position.x = fingertips[i][0]
            marker.pose.position.y = fingertips[i][1]
            marker.pose.position.z = fingertips[i][2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            color = ColorRGBA()
            if is_curr:
                color.r = 0.0
                color.g = 0.5  # Dark Green
                color.b = 0.0
                color.a = 0.5
            else:
                color.r = 0.0
                color.g = 0.5  # Dark Green
                color.b = 0.0
                color.a = 1.0

            marker.color = color
            markers_msg.markers.append(marker)           

        # Publish the markers
        if is_curr:
            self.curr_grip_force_fingers_pub.publish(markers_msg)
        else:
            self.grip_force_fingers_pub.publish(markers_msg)

##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ros_viz = RosVizInterface()
    parser.add_argument('--rs', action='store_true', help='use realsense camera')
    parser.add_argument('--ft', action='store_true', help='use force torque sensor')
    args = parser.parse_args()

    if args.rs:
        rs = RealSense(select_device=args.device)
        intr = rs.get_camera_intrinsics(CameraType.COLOR)
        print(intr)
        
    if args.ft:
        ft_obj = FTCapture()
        calibrate_ft(ft_obj)
        offset = get_ft_calibration()
        cam_rot = ft_to_cam_rotation()

    while True:
        if args.rs:
            color_image, depth_image = rs.get_rgbd_image()
            disp_image = rs.display_rgbd_image(color_image, depth_image)
            pcd = rs.get_point_cloud(color_image, depth_image)
            ros_viz.publish_pcd(pcd)
            
        if args.ft:
            ft = ft_obj.get_ft() - offset
            force_vec = ft[:3]@cam_rot
            ros_viz.publish_wrist_force(force_vec, [0,0,0.2], scale=0.1)
            print(force_vec)

        ros_viz.publish_wrist_force([3,0,1], [0,0,0.2])
        ros_viz.publish_fingertips([[1,0,0], [0,0,0.1]])
        # Show images
        ros_viz.publish_grip_force(3.5, [0,0,0.2])
        ros_viz.publish_grip_force_with_fingertips(
            10.5,
            [[-0.3, 0, 0.2], [0.3, 0, 0.2]]
        )
        time.sleep(1)

        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', disp_image)
        # cv2.waitKey(1)
