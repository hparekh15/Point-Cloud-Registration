import pyrealsense2 as rs
import numpy as np
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    counter = 4

    while True:
        # Wait for Enter key
        input("Press Enter to capture point cloud...")

        frames = pipeline.wait_for_frames() # Get frames to get depth and color from the camera
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

      
        depth_image = np.asanyarray(depth_frame.get_data()) # Convert depth frame to numpy array

        # Create point cloud
        points = rs.pointcloud()
        points.map_to(color_frame)
        point_cloud = points.calculate(depth_frame)

        
        point_cloud_array = np.asanyarray(point_cloud.get_vertices()) # Convert point cloud to numpy array

        # Print number of points in point cloud
        print("Point cloud", counter, "- Number of points:", len(point_cloud_array))

        # Save point cloud to PLY file
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_array)
        o3d.io.write_point_cloud(f"point_cloud_{counter}.ply", pcd)

        # Save point cloud to XYZ file
        # np.savetxt(f"point_cloud_{counter}.xyz", point_cloud_array, delimiter=" ", fmt="%f")

        counter += 1  # Increment the counter

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
