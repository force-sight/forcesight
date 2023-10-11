#!/usr/bin/env python3

import cv2
import numpy as np
from aruco_detect import ArucoPoseEstimator, find_contact_markers
from realsense_utils import DefaultIntrinsic

import rospy
import tf
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge



##############################################################################

class ImageSubscriber:
    def __init__(
                self,
                aruco_pose_est,
                frame_ref="camera_depth_optical_frame",
                contact_callback=None
            ):
        """Image subscriber for aruco pose estimation

        Args:
            aruco_pose_est (_type_): _description_
            frame_ref (str, optional): _description_. Defaults to "camera_depth_optical_frame".
            contact_callback (_type_, optional): _description_. Defaults to None.
        """
        self.bridge = CvBridge()
        self.image_sub = \
            rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.marker_pub = \
            rospy.Publisher("/contact_marker", Marker, queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.aruco_pose_est = aruco_pose_est
        self.ref_frame = frame_ref
        self.contact_callback = contact_callback

    def image_callback(self, data):           
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            m_list = self.aruco_pose_est.detect(cv_image, viz=True)
            
            for m in m_list:
                # print(f"m{m.id}", m.trans, m.rot)

                # Publish the transform to /tf
                self.tf_broadcaster.sendTransform(
                        (m.trans[0], m.trans[1], m.trans[2]),
                        tf.transformations.quaternion_from_euler(
                            m.rot[0], m.rot[1], m.rot[2]),
                        rospy.Time.now(),
                        f"m{m.id}",
                        self.ref_frame
                    )

            if len(m_list) == 2:
                coor1, coor2 = find_contact_markers(m_list[0], m_list[1])
                self.publish_marker_msg([coor1, coor2])
                if self.contact_callback:
                    self.contact_callback(coor1, coor2)

        except Exception as e:
            print(e)
        # else:
            # Process the cv_image here
            # cv2.imshow("Image", cv_image)
        cv2.waitKey(1)
    
    def publish_marker_msg(self, coors, size=0.01, is_point=True):
        marker = Marker()
        marker.header.frame_id = self.ref_frame
        marker.action = Marker.ADD  # set the marker action to ADD
        marker.color.a = 0.8  # set the alpha
        marker.color.r = 1.0
        marker.color.b = 1.0

        if is_point:
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
        else:
            # TODO: use cylinder to represent the contact
            marker.type = Marker.CYLINDER  # set the marker type to CYLINDER
            marker.scale.x = size  # set the radius of the cylinder
            marker.scale.y = size
            # use the dist between two markers as the height of the cylinder
            marker.scale.z = np.linalg.norm(coors[0] - coors[1])
                    
            start = coors[0]
            end = coors[1]
            center = Point()
            center.x = (start[0] + end[0])/2
            center.y = (start[1] + end[1])/2
            center.z = (start[2] + end[2])/2   
            # covert vector to quaternion
            vector = end - start
            quat = tf.transformations.quaternion_about_axis(0, vector)
            marker.pose.position = center
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]

        self.marker_pub.publish(marker)

##############################################################################

if __name__ == "__main__":
    default_intr = DefaultIntrinsic()
    cam_mat = default_intr.cam_mat()
    cam_dist = default_intr.cam_dist()

    def callback(p1, p2):
        print("received callback", p1, p2)

    aruco_pose_est = ArucoPoseEstimator(
        cam_mat, cam_dist, marker_size=0.0155, valid_ids=[11, 12])

    rospy.init_node('gripper_pose_estimator', anonymous=True)
    image_subscriber = ImageSubscriber(aruco_pose_est, contact_callback=callback)
    rospy.spin()

    # Release the video capture object and close all windows
    cv2.destroyAllWindows()
