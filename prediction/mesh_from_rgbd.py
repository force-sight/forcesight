#!pip install torch opencv-python Pillowimport open3d as o3d

import numpy as np
import open3d as o3d

def fit_mesh(color_image, depth_image):
    # Create color and depth Open3D images
    color = o3d.geometry.Image(color_image)
    depth = o3d.geometry.Image(depth_image)

    # Create an RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )

    # Define camera intrinsic parameters
    fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5
    intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, fx, fy, cx, cy)

    # Generate point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # Load the STL mesh
    mesh = o3d.io.read_triangle_mesh("assets/gripper_fingertip.STL")
    mesh.translate(-mesh.get_center())  # Center the mesh

    # Calculate the average depth value from the depth image
    depth_array = np.asarray(depth_image)
    avg_depth = np.mean(depth_array[depth_array > 0])  # Exclude zero values

    # Compute the dynamic scale factor
    # You can adjust the scaling_constant based on your object size and camera setup
    scaling_constant = 0.001
    scale_factor = scaling_constant * avg_depth

    # Scale the mesh based on the scale factor
    mesh.scale(scale_factor, mesh.get_center())

    # Create a PointCloud object from the mesh vertices
    mesh_pcd = o3d.geometry.PointCloud()
    mesh_pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    # Initial transformation can be an identity matrix if you don't have an initial estimate
    initial_transform = np.identity(4)

    # Apply ICP to refine the transformation
    threshold = 0.01
    trans_icp = o3d.pipelines.registration.registration_icp(
        mesh_pcd, pcd, threshold, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ).transformation

    # Apply the transformation to the mesh
    mesh.transform(trans_icp)

    # Turning the mesh orange
    mesh.paint_uniform_color([1, 0.706, 0])

    # Visualize the aligned mesh and point cloud
    o3d.visualization.draw_geometries([pcd, mesh])

