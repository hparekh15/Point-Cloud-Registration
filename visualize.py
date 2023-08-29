import open3d as o3d
# Load pcd file from disk
workspace = o3d.io.read_point_cloud("Test Data 1/workspace_pc.pcd")
cropped_workspace = o3d.io.read_point_cloud("Test Data 1/cropped_workspace_pc.pcd")

# Initialize a coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])

# Visualize the point cloud
# o3d.visualization.draw_geometries([workspace, coordinate_frame]) # For complete workspace
o3d.visualization.draw_geometries([workspace, coordinate_frame]) # For cropped workspace, uncomment this line and comment the above line