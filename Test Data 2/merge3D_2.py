import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# Initialize coordinate frame at origin:
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])

# Read point clouds:
pc_0 = o3d.io.read_point_cloud("Test Data 2\pcd_0.xyz", format='xyz')
pc_1 = o3d.io.read_point_cloud("Test Data 2\pcd_1.xyz", format='xyz')
pc_2 = o3d.io.read_point_cloud("Test Data 2\pcd_2.xyz", format='xyz')
pc_3 = o3d.io.read_point_cloud("Test Data 2\pcd_3.xyz", format='xyz')
pc_4 = o3d.io.read_point_cloud("Test Data 2\pcd_4.xyz", format='xyz')


# EE Pose wrt Franka Base, input fromat (x, y, z, qx, qy, qz, qw):
pose_0 = np.array([0.4388440032543044, -0.0075462174011609465, 0.4500227169578591, 0.9984167281317976, 0.013601041358285134, 0.054512975713610266, -0.001603105833463662])
pose_1 = np.array([0.593014982208658, -0.01440953865760553, 0.47093945587862895, 0.9966198132937542, 0.023327084766205035, -0.06750266719144281, -0.0405384860821075])
pose_2 = np.array([0.32951222364764665, -0.013353048925014155, 0.46197756245136895, 0.991296372296191, 0.0471032671125117, 0.12287748099797435, -0.003014132994755392])
pose_3 = np.array([0.41916590634776996, 0.27148212380121334, 0.4872046957739613, 0.978732990427817, 0.0949678428991891, 0.0629274282139764, 0.1705817177364905])
pose_4 = np.array([0.4628885303619983, -0.30476017841252845, 0.4869179383359469, 0.9855881122065405, 0.012462907015585067, -0.00924378778291739, -0.1684354100116081])

# Camera Pose wrt EE:
cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])


# Downsample the point clouds:
voxel_size = 0.01
pcd_0 = pc_0.voxel_down_sample(voxel_size=voxel_size)
pcd_1 = pc_1.voxel_down_sample(voxel_size=voxel_size)
pcd_2 = pc_2.voxel_down_sample(voxel_size=voxel_size)
pcd_3 = pc_3.voxel_down_sample(voxel_size=voxel_size)
pcd_4 = pc_4.voxel_down_sample(voxel_size=voxel_size)


# Color the point clouds:                              
pcd_0.paint_uniform_color([1, 0, 0]) # red
pcd_1.paint_uniform_color([0, 1, 0]) # green
pcd_2.paint_uniform_color([0, 0, 1]) # blue
pcd_3.paint_uniform_color([1, 1, 0]) # yellow
pcd_4.paint_uniform_color([1, 0, 1]) # pink

# Visazlize point clouds:
# o3d.visualization.draw_geometries([pcd_0, pcd_1, coordinate_frame])

def pose_to_trans_matrix(pose):
    """
    returns transformation matrix from pose
    ---
    input pose: [x, y, z, qx, qy, qz, qw]
    ---
    output H: 4x4 transformation matrix
    """
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


# Camera Calibration:
T_ec = pose_to_trans_matrix(cam2ee_pose)

# Transform point clouds to Franka Base Frame and register:
# H_fc = H_fe * H_ec
# P_f = (H_fc) * P_c 
T_fe0 = pose_to_trans_matrix(pose_0)
T_f0 = (T_fe0.dot(T_ec))
pcd_0.transform(T_f0)

T_fe1 = pose_to_trans_matrix(pose_1)
T_f1 = (T_fe1.dot(T_ec))
reg_p2p_f1 = o3d.pipelines.registration.registration_icp(pcd_1, pcd_0, 0.02, T_f1, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_1.transform(reg_p2p_f1.transformation)

T_fe2 = pose_to_trans_matrix(pose_2)
T_f2 = (T_fe2.dot(T_ec))
reg_p2p_f2 = o3d.pipelines.registration.registration_icp(pcd_2, pcd_0, 0.02, T_f2, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_2.transform(reg_p2p_f2.transformation)

T_fe3 = pose_to_trans_matrix(pose_3)
T_f3 = (T_fe3.dot(T_ec))
reg_p2p_f3 = o3d.pipelines.registration.registration_icp(pcd_3, pcd_0, 0.02, T_f3, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_3.transform(reg_p2p_f3.transformation)

T_fe4 = pose_to_trans_matrix(pose_4)
T_f4 = (T_fe4.dot(T_ec))
reg_p2p_f4 = o3d.pipelines.registration.registration_icp(pcd_4, pcd_0, 0.02, T_f4, o3d.pipelines.registration.TransformationEstimationPointToPoint())
pcd_4.transform(reg_p2p_f4.transformation)

# o3d.visualization.draw_geometries([pcd_0, pcd_1, pcd_2, pcd_3, pcd_4, coordinate_frame])

# Merge point clouds:
print("Merge point clouds")
pcd_merged = pcd_0 + pcd_1 +  pcd_2 + pcd_3 + pcd_4
pcd_merged = pcd_merged.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([pcd_merged, coordinate_frame])

# Save merged point cloud locally
# o3d.io.write_point_cloud("workspace_pc.pcd", pcd_merged) 

# Crop merged point cloud to workspace
print("Crop merged point cloud to workspace")
workspace = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -2, -0.5), max_bound=(2, 2, 1.5))
cropped_pcd = pcd_merged.crop(workspace)
o3d.visualization.draw_geometries([cropped_pcd, coordinate_frame])

# Save croppped point cloud locally
# o3d.io.write_point_cloud("cropped_workspace_pc.pcd", cropped_pcd) 
