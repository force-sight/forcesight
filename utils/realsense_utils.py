#!/usr/bin/env python3

import numpy as np
import cv2
import pyrealsense2 as rs
import open3d as o3d
import time
from typing import Any, Tuple, List
import argparse
# from mesh_from_rgbd import *

##############################################################################

class DefaultIntrinsic:
    """
    This is the default intrinsic from the D405 realseense rgb camera with 848
    """
    ppx = 415.507537841797
    ppy = 237.871643066406
    fx = 431.125
    fy = 430.667
    coeffs = [-0.053341, 0.0545209, 0.000824648, 0.000749805, -0.0171459]

    # https://github.com/IntelRealSense/librealsense/issues/3473#issuecomment-474637827
    depth_scale = 9.9999e-05

    def cam_mat(self):
        return camera_matrix(self)

    def cam_dist(self):
        return fisheye_distortion(self)

class Intrinsic640(DefaultIntrinsic):
    ppx = 311.508
    ppy =  237.872
    fx = 431.125
    fy = 430.667
    coeffs = [-0.053341, 0.0545209, 0.000824648, 0.000749805, -0.0171459]

def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx, 0, intrinsics.ppx],
                     [0, intrinsics.fy, intrinsics.ppy],
                     [0,             0,              1]])


def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

##############################################################################

def get_point_cloud(color_image, depth_image, intrinsics):
    """creating point cloud in open3d"""
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Create an Open3D camera intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)

    # print('color_image dtype', color_image.dtype)
    # print('depth_image dtype', depth_image.dtype)

    # check if rgb imge is dtype uint8 else convert
    if color_image.dtype == np.float32:
        print("color image is not uint8")
        color_image = color_image*255
        color_image = color_image.astype(np.uint8)

    # check if rgb imge is dtype uint16 else convert
    if depth_image.dtype == np.float32:
        # convert to uint16
        print("depth image is not uint16")
        depth_image = depth_image*65535.0
        depth_image = depth_image.astype(np.uint16)

    color_image = o3d.geometry.Image(color_image)
    depth_image = o3d.geometry.Image(depth_image)

    # Create an Open3D RGBDImage from the color and depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=10000.0,
        depth_trunc=2.0, convert_rgb_to_intensity=False)

    # Create a point cloud from the RGBD image and camera intrinsic parameters
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

def pcd_from_rgbd(rgb, depth, intr):
    color_image = cv2.imread(rgb)
    depth_image = cv2.imread(depth, cv2.IMREAD_ANYDEPTH)

    # show both rgb and depth images
    # img = np.hstack((color_image, depth_image)
    # cv2.imshow('image', img)
    depth_image = depth_image.astype(np.float32)
    depth_image = depth_image.astype(np.int16)

    # Test resize, INTER_NEAREST is important to keep the depth values
    depth_image = cv2.resize(depth_image, (320, 240), interpolation=cv2.INTER_NEAREST)
    depth_image = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_NEAREST)

    return get_point_cloud(color_image, depth_image, intr)

##############################################################################

class PCDViewer:
    def __init__(self, skip_frames=10, blocking=False):
        """
        pcd viewer to display point cloud
        :arg skip_frames: number of frames to skip, for live display
        :arg blocking: if true, will block until window is closed, this will
                        enable mouse control when blocking
        """
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        # Set a custom viewpoint: TODO: edit this, viewing angle
        # rotate around z axis
        ctr = self.vis.get_view_control()
        ctr.set_lookat([0, 0, 0.2])
        ctr.set_front([0, -1, -1])
        ctr.set_up([0, -1, 0])
        self.skip = skip_frames
        self.counter = 0
        self.blocking = blocking
        # self.pcd = o3d.geometry.PointCloud()
        # self.vis.add_geometry(self.pcd)
    
    def display(self, points, markers=[], ignore_skip=False):
        """
        display point cloud and fingertip markers
        ignore skip will display the frame regardless of skip
        """
        if ignore_skip or self.blocking:
            self.counter = 0

        _markers = []
        for pos in markers:
            marker_pos = pos.reshape(3) # np.array([0.5, 0.5, 0.5])
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            marker.paint_uniform_color([1, 0, 0])  # Set the color to red
            marker.translate(marker_pos)
            _markers.append(marker)

        if self.counter%self.skip == 0:
            # To get viz camera control of the previous frame, then reset the view
            # since add geo will reset it:
            # https://github.com/isl-org/Open3D/issues/2264
            ctr  = self.vis.get_view_control()
            view_param = ctr.convert_to_pinhole_camera_parameters()
            self.vis.clear_geometries()
            self.vis.add_geometry(points)
            for m in _markers:
                self.vis.add_geometry(m)

            ctr.convert_from_pinhole_camera_parameters(view_param)
            self.counter = 0
        self.counter += 1

        # print number of points
        print("Number of points: {}".format(len(points.points)))

        if self.blocking:
            self.vis.run()
        else:
            self.vis.poll_events()
            self.vis.update_renderer()

    def __del__(self):
        self.vis.destroy_window()


def display_point_cloud(points, markers=[]):
    # point cloud visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.clear_geometries()
    vis.add_geometry(points)
    print("Number of points: {}".format(len(points.points)))
    
    for pos in markers:
        marker_pos = pos.reshape(3) # np.array([0.5, 0.5, 0.5])
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        marker.paint_uniform_color([1, 0, 0])  # Set the color to red
        marker.translate(marker_pos)
        vis.add_geometry(marker)

    # Set a custom viewpoint: TODO: edit this, viewing angle
    # rotate around z axis
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0.2])
    ctr.set_front([0, -0.5, -0.5])
    ctr.set_up([0, -1, 0])

    # vis.run()
    while vis.poll_events():
        vis.update_renderer()
    vis.destroy_window()

##############################################################################

class CameraType:
    COLOR = rs.stream.color
    DEPTH = rs.stream.depth


##############################################################################
class RealSense:
    def __init__(self, select_device="127122270519", view=False, auto_expose=True): #127122270519
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        print("selecting device: {}".format(select_device))
        if select_device is not None:
            self.config.enable_device(select_device)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        cfg = self.pipeline.start(self.config)

        color_sensor = cfg.get_device().query_sensors()[0]
        color_sensor.set_option(rs.option.enable_auto_exposure, auto_expose)

        # getting camera intrinsics
        _depth_profile = cfg.get_stream(rs.stream.depth)
        self.depth_intr = \
            _depth_profile.as_video_stream_profile().get_intrinsics()
        _rgb_profile = cfg.get_stream(rs.stream.color)
        self.rgb_intr = \
            _rgb_profile.as_video_stream_profile().get_intrinsics()

        self.first_frame_time = 0
        self.current_frame_time = 0
        self.frame_count = 0
        self.view = view

    def get_rgbd_image(self):
        """Return a pair of color and depth frame from the realsense camera."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        self.current_frame_time = time.time()

        if self.first_frame_time == 0:
            self.first_frame_time = self.current_frame_time

        if not depth_frame or not color_frame:
            print("frame {} was bad".format(self.frame_count))
            return None
        
        self.frame_count += 1

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) # depth in mm
        color_image = np.asanyarray(color_frame.get_data())

        # print('depth_min', depth_image.min())
        # print('depth_mean', depth_image.mean())
        # print('depth_max', depth_image.max())

        return color_image, depth_image

    def display_rgbd_image(self, color_image, depth_image):
        """Display color and depth images."""

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        return images
    
    def get_point_cloud(self, color_image, depth_image):
        """creating point cloud in open3d"""
        # TODO: check if intrinsics should be from depth or color camera
        return get_point_cloud(color_image, depth_image, self.depth_intr)

    def get_camera_intrinsics(
        self, type: CameraType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the camera calibration parameters from the realsense camera
        Ref: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/t265_stereo.py
        :return: camera matrix and distortion coefficients
        """
        if type == CameraType.COLOR:
            mat = camera_matrix(self.rgb_intr)
            dist = fisheye_distortion(self.rgb_intr)
        elif type == CameraType.DEPTH:
            mat = camera_matrix(self.depth_intr)
            dist = fisheye_distortion(self.depth_intr)
        return mat, dist

    def get_frame(self, type: CameraType = CameraType.COLOR):
        """
        Get a frame from the realsense camera
        """
        frames = self.pipeline.wait_for_frames()
        if type == CameraType.COLOR:
            color_frame = frames.get_color_frame()
            return np.asanyarray(color_frame.get_data())
        else:
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            return depth_colormap

    # detruct the wrapper
    def __del__(self):
        self.pipeline.stop()

##############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud', action="store_true", help='view pointcloud')
    parser.add_argument('-d', '--device', type=str, default=None, help='device to use')
    parser.add_argument('--rgb', type=str, default=None, help='path to rgb image')
    parser.add_argument('--depth', type=str, default=None, help='path to depth image')
    args = parser.parse_args()

    if args.rgb is not None and args.depth is not None:
        print('viewing pointcloud from images')
        
        pcd = pcd_from_rgbd(args.rgb, args.depth, Intrinsic640())
        pcd_vis = PCDViewer(blocking=True)
        pcd_vis.display(pcd)

        # save the point cloud
        o3d.io.write_point_cloud('viz_pcd.ply', pcd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    rs = RealSense(select_device=args.device)
    intr = rs.get_camera_intrinsics(CameraType.COLOR)
    print(intr)

    pcd_vis = PCDViewer()
    while True:
        color_image, depth_image = rs.get_rgbd_image()
        disp_image = rs.display_rgbd_image(color_image, depth_image)
        if args.cloud:
            print('depth_image dtype', depth_image.dtype)

            pcd = rs.get_point_cloud(color_image, depth_image)
            pcd_vis.display(pcd)
            
            # save depth and color images
            # cv2.imwrite('color.png', color_image)
            depth_image = depth_image.astype('uint16')
            # cv2.imwrite('depth.png', depth_image)

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', disp_image)
        cv2.waitKey(1)
