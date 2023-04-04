import open3d as o3d
import numpy as np
import rospy #INSTALL ROS NOETIC
import ros_numpy

num_perspectives = 5

# Load ROS point cloud data in numpy format
ros_pointcloud = rospy.wait_for_message("/pointcloud_topic", PointCloud2) #need to edit based on which perspective we want
pointcloud_np = ros_numpy.point_cloud2.pointcloud2_to_array(ros_pointcloud)
points_np = ros_numpy.numpify(ros_pointcloud)
# Convert numpy array to open3d point cloud format
points_o3d = o3d.geometry.PointCloud()
points_o3d.points = o3d.utility.Vector3dVector(points_np[:, :3])

# Load the first point cloud
pcd_list = [o3d.io.read_point_cloud("point_cloud_1.pcd")]

# Load the rotation and translation matrices for all other perspectives from Robot
R_list = []
T_list = []

for i in range(2, num_perspectives+1):
    R = np.load("rotation_matrix_{}_{}.npy".format(i-1, i))
    T = np.load("translation_vector_{}_{}.npy".format(i-1, i))
    R_list.append(R)
    T_list.append(T)

# Register each consecutive pair of point clouds using Iterative Close Point Alg
for i in range(1, num_perspectives):
    pcd_i = o3d.io.read_point_cloud("point_cloud_{}.pcd".format(i+1))
    pcd_j = pcd_list[i-1]

    R_ij = R_list[i-1]
    T_ij = T_list[i-1]

    pcd_i.transform(R_ij, T_ij)

    icp_result = o3d.pipelines.registration.registration_icp(pcd_i, pcd_j, max_correspondence_distance=0.1)

    pcd_i.transform(icp_result.transformation)

    pcd_list.append(pcd_i)

# Merge all point clouds into a single point cloud
merged_pcd = pcd_list[0]
for pcd in pcd_list[1:]:
    merged_pcd += pcd

# Refine the alignment between all point clouds using global registration using RANSAC
threshold = 0.2
trans_init = np.identity(4)

reg_p2p = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    merged_pcd, merged_pcd, o3d.pipelines.registration.Feature(), o3d.pipelines.registration.Feature(),
    threshold, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(threshold)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
)

merged_pcd.transform(reg_p2p.transformation)

# Perform surface reconstruction to create a continuous 3D map
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merged_pcd, depth=8)

# Visualize the 3D map
o3d.visualization.draw_geometries([mesh])
