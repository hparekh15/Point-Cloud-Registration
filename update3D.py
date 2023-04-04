import open3d as o3d
import numpy as np
import rospy #INSTALL ROS NOETIC
import ros_numpy

# Load the existing 3D map
existing_map = o3d.io.read_point_cloud("existing_map.pcd")

# Load the new perspective and the corresponding rotation and translation matrices

# Load ROS point cloud data in numpy format
ros_pointcloud = rospy.wait_for_message("/pointcloud_topic", PointCloud2) #need to edit based on which perspective we want
pointcloud_np = ros_numpy.point_cloud2.pointcloud2_to_array(ros_pointcloud)
points_np = ros_numpy.numpify(ros_pointcloud)
# Convert numpy array to open3d point cloud format
points_o3d = o3d.geometry.PointCloud()
points_o3d.points = o3d.utility.Vector3dVector(points_np[:, :3])

new_perspective = o3d.io.read_point_cloud("points_o3d.pcd")
R_new = np.load("rotation_matrix_new.npy")
T_new = np.load("translation_vector_new.npy")

# Transform the new perspective to align with the existing map
new_perspective.transform(R_new, T_new)

# Perform registration between the new perspective and the existing map
icp_result = o3d.pipelines.registration.registration_icp(new_perspective, existing_map, max_correspondence_distance=0.1)

# Transform the new perspective to the new coordinate system
new_perspective.transform(icp_result.transformation)

# Update the existing map with the new perspective
existing_map += new_perspective

# Refine the alignment between all point clouds using global registration
threshold = 0.2
trans_init = np.identity(4)

reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    existing_map, existing_map, o3d.pipelines.registration.Feature(), o3d.pipelines.registration.Feature(),
    threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)

existing_map.transform(reg_p2p.transformation)

# Perform surface reconstruction to create a continuous 3D map
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(existing_map, depth=8)

# Filter cloud to workspace/reachability of arm
# Add general filter for 2x2x2m from world frame

# Visualize the updated 3D map
o3d.visualization.draw_geometries([mesh])
