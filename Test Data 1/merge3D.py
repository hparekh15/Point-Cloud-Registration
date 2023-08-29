import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
# from math import cos, sin

coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])

pc_0 = o3d.io.read_point_cloud("Test Data 1\point_cloud_1.pcd")
pc_1 = o3d.io.read_point_cloud("Test Data 1\point_cloud_2.pcd")
pc_2 = o3d.io.read_point_cloud("Test Data 1\point_cloud_3.pcd")

# T wrt Franka base, R wrt pose 1, input fromat (x, y, z, qx, qy, qz, qw):
pose_0 = np.array([0.45, 0.3, 0.4, 0.925, 0.0, 0.0, 0.381])
pose_1 = np.array([0.45, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0])
pose_2 = np.array([0.45, -0.3, 0.4, -0.925, 0.0, 0.0, 0.381])

cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])

# Downsample the point clouds
voxel_size = 0.01
pcd_0 = pc_0.voxel_down_sample(voxel_size=voxel_size)
pcd_1 = pc_1.voxel_down_sample(voxel_size=voxel_size)
pcd_2 = pc_2.voxel_down_sample(voxel_size=voxel_size)

# Color the point clouds RGB                               
pcd_0.paint_uniform_color([1, 0, 0])
pcd_1.paint_uniform_color([0, 1, 0])
pcd_2.paint_uniform_color([0, 0, 1])

# Visazlize point clouds:
# o3d.visualization.draw_geometries([pcd_0, pcd_1, pcd_2])

# QUAT / RPY Method
def pose_to_trans_matrix(pose):
    x, y, z, qx, qy, qz, qw = pose
    
    # Rx = np.array([[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]])
    # Ry = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])
    # Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
    # R = Rz.dot(Ry).dot(Rx)
    
    r = R.from_quat([qx, qy, qz, qw])
    Rot = r.as_matrix()
    trans = np.array([[x], [y], [z]])
    
    H = np.block([[Rot, trans], [np.zeros((1, 3)), 1]])
    
    H_inv = np.block([[Rot.T, -Rot.T.dot(trans)], [np.zeros((1, 3)), 1]])
    
    return H


# Camera frame in EE Frame via callibration
# T_ec = np.eye(4)
# Rz = R.from_euler('z', np.pi/2).as_matrix()
# Rx = R.from_euler('x', np.pi).as_matrix()
# R_ec = Rx.dot(Rz)
# T_ec[:3, :3] = R_ec
# T_ec[0, 3] = 0.02
# T_ec[1, 3] = 0.0
# T_ec[2, 3] = 0.0

# Transform point clouds to EE frame
T_ec = pose_to_trans_matrix(cam2ee_pose)
# pcd_0.transform(T_ec)
# pcd_1.transform(T_ec)
# pcd_2.transform(T_ec)

# o3d.visualization.draw_geometries([pcd_0, pcd_1, pcd_2, coord])


# Transform point clouds to Franka Base frame
T_fe0 = pose_to_trans_matrix(pose_0)
new_T_fe0 = np.eye(4)
new_T_fe0[:3, :3] = T_fe0[:3, :3]
# pcd_0.transform(new_T_fe0.T)

T_fe1 = pose_to_trans_matrix(pose_1)
new_T_fe1 = np.eye(4)
new_T_fe1[:3, :3] = T_fe1[:3, :3]
# pcd_1.transform(new_T_fe1.T)

T_fe2 = pose_to_trans_matrix(pose_2)
new_T_fe2 = np.eye(4)
new_T_fe2[:3, :3] = T_fe2[:3, :3]
# pcd_2.transform(new_T_fe2.T)

# o3d.visualization.draw_geometries([pcd_0, pcd_1, pcd_2, coord])



# MANUAL Method
T01 = np.eye(4) # Manual transformation matrix between pcd0 and pcd1
T01[:3, :3] = pcd_0.get_rotation_matrix_from_xyz((0, -(np.pi/6) - np.pi/36, 0)) # Rotate 35 degrees around neg y-axis
T01[0, 3] = -0.3 # Translate 0.3 meters along x-axis
# trans_pcd_0 = copy.deepcopy(pcd_0).transform(T01)


T21 = np.eye(4) # Manual transformation matrix between pcd2 and pcd1
T21[:3, :3] = pcd_2.get_rotation_matrix_from_xyz((0, (np.pi/4), 0)) #Rotate 45 degrees around y-axis
T21[0, 3] = 0.3 # Translate 0.3 meters along x-axis
# trans_pcd_2 = copy.deepcopy(pcd_2).transform(T21)


# print("Display all point clouds")
# o3d.visualization.draw_geometries([pcd_1, trans_pcd_0, trans_pcd_2, coord])


# print("Applying point-to-point ICP to pcd_0")
thresh = 0.02
reg_p2p_01 = o3d.pipelines.registration.registration_icp( pcd_0, pcd_1, thresh, T01, o3d.pipelines.registration.TransformationEstimationPointToPoint())
trans_pcd_0 = copy.deepcopy(pcd_0).transform(reg_p2p_01.transformation)

# print("Applying point-to-point ICP to pcd_2")
reg_p2p_21 = o3d.pipelines.registration.registration_icp( pcd_2, pcd_1, thresh, T21, o3d.pipelines.registration.TransformationEstimationPointToPoint())
trans_pcd_2 = copy.deepcopy(pcd_2).transform(reg_p2p_21.transformation)

print("Merge point clouds & Display")
pcd_merged = pcd_1 +  trans_pcd_0 + trans_pcd_2
pcd_merged = pcd_merged.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([pcd_merged, coord])

# o3d.io.write_point_cloud("workspace_pc.pcd", pcd_merged) # Save merged point cloud locally

# Crop merged point cloud to workspace
# min_bound = pcd_merged.get_min_bound() #[-5.14150218 -2.35588855 -3.57408645]
# max_bound = pcd_merged.get_max_bound() # [2.13656862 2.07392709 0.06327091]
# print("Crop merged point cloud to workspace")
workspace = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(1, 1, 0.05))
cropped_pcd = pcd_merged.crop(workspace)
o3d.visualization.draw_geometries([cropped_pcd, coord])

# o3d.io.write_point_cloud("cropped_workspace_pc.pcd", cropped_pcd) # Save merged point cloud locally