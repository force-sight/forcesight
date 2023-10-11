#!/usr/bin/env python3

import numpy as np
import cv2
import cv2.aruco as aruco
from typing import Any, Tuple, List
import pyrealsense2 as rs
import numpy as np
import argparse
from utils.realsense_utils import RealSense, CameraType

##############################################################################

def find_contact_markers(v1, v2, offset_from_marker=0.025):
    """
    Find the contact point of the two markers on the stretch gripper
    """
    plane_normal = np.cross(v1, v2)
    contact_vec1 = np.cross(v1, plane_normal)
    contact_vec2 = np.cross(v2, plane_normal)

    # convert to unit vector then scale by offset_from_marker
    contact_vec1 = contact_vec1*offset_from_marker/np.linalg.norm(contact_vec1)
    contact_vec2 = contact_vec2*offset_from_marker/np.linalg.norm(contact_vec2)

    # calculate contact points (left and right)
    contact_p1 = v1 - contact_vec1
    contact_p2 = v2 + contact_vec2
    # print("contact_p1", contact_p1)
    # print("contact_p2", contact_p2)
    return contact_p1, contact_p2


##############################################################################

def get_camera_calib(calibration_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the camera calibration parameters from the calibration file
    :param calibration_file: path to the calibration file
    :return: camera matrix and distortion coefficients
    """
    calibration_params = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    cam_mat = calibration_params.getNode("Camera_Matrix").mat()
    cam_dist = calibration_params.getNode("Distortion_Coefficients").mat()
    print(cam_mat, cam_dist)
    assert cam_mat is not None and cam_dist is not None, \
        "Camera calibration file is not valid"
    return cam_mat, cam_dist

##############################################################################

class MarkerPose:
    id: int 
    trans: np.ndarray # xyz
    rot: np.ndarray # rpy

class ArucoPoseEstimator:
    def __init__(
                self,
                camera_matrix,
                dist_coeffs,
                marker_size=0.016,
                valid_ids=None,
            ) -> None:
        """
        This class is used to detect aruco markers from a frame and
        return the pose of the marker
        :valid_ids: a list of valid ids to detect, if None, detect all ids
        """
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()  # TODO: tune these params
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_size = marker_size  # size of the aruco marker in meters
        self.valid_ids = valid_ids

    def detect(self, frame, viz=False) -> List[MarkerPose]:
        """
        detect aruco marker from frame and return the pose of the marker
        """
        corners, ids, rejected = aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params)

        if viz:
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.imshow("frame", frame)

        if self.valid_ids is not None:
            aruco_mask = np.isin(ids, self.valid_ids)
            aruco_mask = np.squeeze(aruco_mask)
            if aruco_mask.size > 0:
                ids = ids[aruco_mask]
                corners = np.array(corners)[aruco_mask]
                assert len(ids) == len(corners)

        m_list = []
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_size,
                self.camera_matrix, self.dist_coeffs)
            for i in range(len(ids)):
                m = MarkerPose()
                m.id = ids[i][0]
                m.trans = tvecs[i][0]
                m.rot = rvecs[i][0]
                m_list.append(m)
        return m_list
    
    def get_fingertip_poses(self, rgb_image):
        """
        get fingertip poses with aruco, return as np array
        For our system, ID 11 is left fingertip, ID 12 is right fingertip
        """
        arucos = self.detect(rgb_image)

        ids = [x.id for x in arucos]

        if 11 in ids and 12 in ids:
            left_fingertip = np.array(arucos[ids.index(11)].trans)
            right_fingertip = np.array(arucos[ids.index(12)].trans)    
            return left_fingertip, right_fingertip
        return None

##############################################################################


if __name__ == "__main__":

    # init argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load",
                        help="load calibration file", action="store_true")
    parser.add_argument("--rs", help="use realsense camera", action="store_true")
    parser.add_argument('--realsense_id', '-rs', type=str, default=None, help='realsense id')
    args = parser.parse_args()

    # init aruco pose estimator

    if args.rs:
        device = RealSense(select_device=args.realsense_id)
        cam_mat, cam_dist = device.get_camera_intrinsics(CameraType.COLOR)
        print(cam_mat, cam_dist)
    else:
       # Initialize the video capture object
        cap = cv2.VideoCapture(2)
        calibration_file = "calibration_file.xml"
        cam_mat, cam_dist = get_camera_calib(calibration_file)

    aruco_pose_estimator = ArucoPoseEstimator(cam_mat, cam_dist)

    counts = {0: 0, 1: 0, 2: 0}

    while True:
        # Capture a frame from the camera
        if args.rs:
            frame = device.get_frame()
            ret = True
        else:
            ret, frame = cap.read()

        if not ret:
            break

        # # Convert the frame to grayscale
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m_list = aruco_pose_estimator.detect(frame, viz=True)
        if len(m_list) == 2 and m_list[0].id == 11 and m_list[1].id == 12:
            coors = find_contact_markers(m_list[0].trans, m_list[1].trans)
            print("fingertips detected: ", coors)

        # Exit if the 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF

        # save frame locally
        if key == ord('q'):
            # cv2.imwrite("last_frame.jpg", frame)
            break
        if key == ord('0'):
            print("save frame 0")
            cv2.imwrite(f"frame_0_{counts[0]}.jpg", frame)
            # TODO: save eef transformation
        elif key == ord('1'):
            print("save frame 1")
            cv2.imwrite(f"frame_1_{counts[1]}.jpg", frame)
            # TODO: save eef transformation and action target transformation
        elif key == ord('2'):
            print("save frame 2")
            cv2.imwrite(f"frame_2_{counts[2]}.jpg", frame)
            # TODO: save eef transformation and action target transformation
        else:
            pass

    cv2.destroyAllWindows()
