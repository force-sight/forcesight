import numpy as np
import cv2
import pyrealsense2 as rs
from prediction.owlvit_seg import OwlViTSeg

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

owlvit = OwlViTSeg()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data()) # depth in mm
        color_image = np.asanyarray(color_frame.get_data())

        print('depth_min', depth_image.min())
        print('depth_mean', depth_image.mean())
        print('depth_max', depth_image.max())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # rotating images 90 degrees ccw
        depth_colormap = np.rot90(depth_colormap)
        color_image = np.rot90(color_image)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        print('color_image dtype', color_image.dtype)
        owlvit.segment(color_image, ['human face'])

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        # # creating point cloud
        # points = rs.pointcloud()
        # points.map_to(color_frame)
        # pointcloud = points.calculate(depth_frame)

        # # displaying point cloud
        # pc = np.asanyarray(pointcloud.get_vertices())
        # pc = pc.view(np.float32).reshape(pc.shape + (-1,))
        # print(pc.shape)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1000.0,  # Set the depth scale according to your depth image format (e.g., 1000.0 for millimeters)
        depth_trunc=3.0,  # Set the depth truncation distance (in meters) for points that are too far away
        convert_rgb_to_intensity=False  # Set to True if you want to convert the RGB values to intensity values
        )

        # Default pinhole camera model:
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )

        # Or, specify your own camera parameters:
        fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5  # Replace these with your actual camera parameters
        width, height = 640, 480  # Replace these with your actual image dimensions
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics
        )

        o3d.visualization.draw_geometries([point_cloud])
        o3d.io.write_point_cloud('output_point_cloud.pcd', point_cloud)


finally:
    # Stop streaming
    pipeline.stop()
