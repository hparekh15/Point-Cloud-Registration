import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
from math import cos, sin

pc_0 = o3d.io.read_point_cloud("point_cloud_1.pcd")
pc_1 = o3d.io.read_point_cloud("point_cloud_2.pcd")
pc_2 = o3d.io.read_point_cloud("point_cloud_3.pcd")
pose_0 = np.array([0.45, 0.3, 0.4, 0.925, 0.0, 0.0, 0.381])
pose_1 = np.array([0.45, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
pose_2 = np.array([0.45, -0.3, 0.4, -0.925, 0.0, 0.0, 0.381])

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])

# Downsample the point clouds
voxel_size = 0.01
pcd_0 = pc_0.voxel_down_sample(voxel_size=voxel_size)
pcd_1 = pc_1.voxel_down_sample(voxel_size=voxel_size)
pcd_2 = pc_2.voxel_down_sample(voxel_size=voxel_size)


# Color the point clouds RGB                               
pcd_0.paint_uniform_color([1, 0, 0])
pcd_1.paint_uniform_color([0, 1, 0])
pcd_2.paint_uniform_color([0, 0, 1])

# QUAT / RPY Method
def pose_to_trans_matrix(pose):
    # Extract position and orientation from the input array
    # x, y, z, roll, pitch, yaw, w = pose
    x, y, z, qx, qy, qz, qw = pose
    
    # Define the rotation matrix
    # Rx = np.array([[1, 0, 0],
    #                [0, cos(roll), -sin(roll)],
    #                [0, sin(roll), cos(roll)]])
    # Ry = np.array([[cos(pitch), 0, sin(pitch)],
    #                [0, 1, 0],
    #                [-sin(pitch), 0, cos(pitch)]])
    # Rz = np.array([[cos(yaw), -sin(yaw), 0],
    #                [sin(yaw), cos(yaw), 0],
    #                [0, 0, 1]])
    # R = Rz.dot(Ry).dot(Rx)
    
    r = R.from_quat([qx, qy, qz, qw])
    Rot = r.as_matrix()
    
    # Define the translation vector
    T = np.array([[x], [y], [z]])
    
    # Combine the rotation matrix and translation vector into a homogeneous transformation matrix
    # T_w_r = np.block([[R, T],
    #                   [np.zeros((1, 3)), 1]])
    
    T_w_r = np.block([[Rot.T, -Rot.T.dot(T)],
                      [np.zeros((1, 3)), 1]])
    
    return T_w_r

T0W = pose_to_trans_matrix(pose_0)
# print("T0W Rotation", T0W[:3, :3])
pcd_0.transform(T0W)

T1W = pose_to_trans_matrix(pose_1)
# print("T1W Rotation", T1W[:3, :3])
pcd_1.transform(T1W)

T2W = pose_to_trans_matrix(pose_2)
# print("T2W Rotation", T2W[:3, :3])
pcd_2.transform(T2W)
o3d.visualization.draw_geometries([ pcd_0, pcd_1, pcd_2, coord])



# MANUAL Method
T01 = np.eye(4) # Manual transformation matrix between pcd2 and pcd1
T01[:3, :3] = pcd_0.get_rotation_matrix_from_xyz((0, -(np.pi/6) - np.pi/36, 0)) # Rotate 35 degrees around neg y-axis
print("Manual 0 to 1", T01[:3, :3])
T01[0, 3] = -0.3 # Translate 0.3 meters along x-axis
trans_pcd_0 = copy.deepcopy(pcd_0).transform(T01)
# o3d.visualization.draw_geometries([pcd_1, trans_pcd_0, coord])

T21 = np.eye(4) # Manual transformation matrix between pcd2 and pcd1
T21[:3, :3] = pcd_2.get_rotation_matrix_from_xyz((0, (np.pi/4), 0)) #Rotate 45 degrees around y-axis
T21[0, 3] = 0.3 # Translate 0.3 meters along x-axis
trans_pcd_2 = copy.deepcopy(pcd_2).transform(T21)
# o3d.visualization.draw_geometries([pcd_1, trans_pcd_2, coord])

# print("Display all point clouds")
# o3d.visualization.draw_geometries([pcd_1, trans_pcd_0, trans_pcd_2, coord])

# print("Merge point clouds & Display")
# pcd_merged = pcd_1 +  trans_pcd_0 + trans_pcd_2
# o3d.visualization.draw_geometries([pcd_merged, coord])

print("Applying point-to-point ICP to pcd_0")
thresh = 0.02
reg_p2p_01 = o3d.pipelines.registration.registration_icp( pcd_0, pcd_1, thresh, T01, o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p_01)
# print("Transformation is:")
# print(reg_p2p_01.transformation)
trans_pcd_0 = copy.deepcopy(pcd_0).transform(reg_p2p_01.transformation)

print("Applying point-to-point ICP to pcd_2")
reg_p2p_21 = o3d.pipelines.registration.registration_icp( pcd_2, pcd_1, thresh, T21, o3d.pipelines.registration.TransformationEstimationPointToPoint())
# print(reg_p2p_21)
# print("Transformation is:")
# print(reg_p2p_21.transformation)
trans_pcd_2 = copy.deepcopy(pcd_2).transform(reg_p2p_21.transformation)


# o3d.visualization.draw_geometries([trans_pcd_2, trans_pcd_0, pcd_1, coord])

