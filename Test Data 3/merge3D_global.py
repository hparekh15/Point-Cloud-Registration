import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy

# Initialize coordinate frame at origin:
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])

# Read point clouds:
pc_0 = o3d.io.read_point_cloud("Test Data 3\pcd_0.xyz", format='xyz')
pc_1 = o3d.io.read_point_cloud("Test Data 3\pcd_1.xyz", format='xyz')
pc_2 = o3d.io.read_point_cloud("Test Data 3\pcd_2.xyz", format='xyz')
pc_3 = o3d.io.read_point_cloud("Test Data 3\pcd_3.xyz", format='xyz')
pc_4 = o3d.io.read_point_cloud("Test Data 3\pcd_4.xyz", format='xyz')

# Color the point clouds:                              
pc_0.paint_uniform_color([1, 0, 0]) # red
pc_1.paint_uniform_color([0, 1, 0]) # green
pc_2.paint_uniform_color([0, 0, 1]) # blue
pc_3.paint_uniform_color([1, 1, 0]) # yellow
pc_4.paint_uniform_color([1, 0, 1]) # pink


# EE Pose wrt Franka Base, input fromat (x, y, z, qx, qy, qz, qw):
pose_0 = np.array([0.4388440032543044, -0.0075462174011609465, 0.4500227169578591, 0.9984167281317976, 0.013601041358285134, 0.054512975713610266, -0.001603105833463662])
pose_1 = np.array([0.593014982208658, -0.01440953865760553, 0.47093945587862895, 0.9966198132937542, 0.023327084766205035, -0.06750266719144281, -0.0405384860821075])
pose_2 = np.array([0.32951222364764665, -0.013353048925014155, 0.46197756245136895, 0.991296372296191, 0.0471032671125117, 0.12287748099797435, -0.003014132994755392])
pose_3 = np.array([0.41916590634776996, 0.27148212380121334, 0.4872046957739613, 0.978732990427817, 0.0949678428991891, 0.0629274282139764, 0.1705817177364905])
pose_4 = np.array([0.4628885303619983, -0.30476017841252845, 0.4869179383359469, 0.9855881122065405, 0.012462907015585067, -0.00924378778291739, -0.1684354100116081])

# Camera Pose wrt EE:
cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])

# Visazlize point clouds:
o3d.visualization.draw_geometries([pc_0, pc_4, coordinate_frame])

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


def transformation_to_franka_base(pose, T_ec):
    T_f = pose_to_trans_matrix(pose) @ T_ec
    return T_f

# Camera frame in EE Frame via callibration:
H_ec = pose_to_trans_matrix(cam2ee_pose)

# Transform point clouds to Franka Base Frame and register:
H_f0 = transformation_to_franka_base(pose_0, H_ec)
# pc_0.transform(H_f0) # TARGET PC

H_f1 = transformation_to_franka_base(pose_1, H_ec)
# pc_1.transform(H_f1)

H_f2 = transformation_to_franka_base(pose_2, H_ec)
# pc_2.transform(H_f2)

H_f3 = transformation_to_franka_base(pose_3, H_ec)
# pc_3.transform(H_f3)

H_f4 = transformation_to_franka_base(pose_4, H_ec)
# pc_4.transform(H_f4) 

# Visazlize point clouds:
# o3d.visualization.draw_geometries([pc_0, pc_4, coordinate_frame])

# Merge point clouds & Display:
# pcd_merged = pcd_0 +  pcd_1 + pcd_2 + pcd_3 + pcd_4
# pcd_merged = pcd_merged.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([pcd_merged, coordinate_frame])

# Save merged point cloud locally:
# o3d.io.write_point_cloud("Test Data 3\workspace_pc.xyz", pcd_merged) 

# Crop merged point cloud to workspace, display & save:
# workspace = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -2, -0.5), max_bound=(2, 2, 1.5))
# cropped_pcd = pcd_merged.crop(workspace)
# o3d.visualization.draw_geometries([cropped_pcd, coordinate_frame])
# o3d.io.write_point_cloud("Test Data 3\cropped_workspace_pc.pcd", cropped_pcd) 

# Remove outliers:
# clean_cloud, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.9)
# cl, ind = cropped_pcd.remove_radius_outlier(nb_points=50, radius=0.1)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    

# display_inlier_outlier(cropped_pcd, ind)
# o3d.visualization.draw_geometries([clean_cloud, coordinate_frame])


# GLOBAL REGISTRATION:

def draw_registration_result(source, target):
    """
    displays registration result 
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def preprocess_point_cloud(pcd, voxel_size):
    """
    returns downsampled point cloud and FPFH feature of point cloud
    """
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute Fast Point Feature Histogram feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, H, voxel_size):
    """
    returns downsampled point clouds and FPFH features of point clouds
    """
    trans_init =np.eye(4)
    # source.transform(trans_init)
    source.transform(H)
    # draw_registration_result(source, target, trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    iter = 100000
    conf = 0.999
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, 
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)], 
        o3d.pipelines.registration.RANSACConvergenceCriteria(iter, conf))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, H, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, H.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

# Global Registration Pipeline:
# Load point clouds:
target = pc_0.transform(H_f0)
source  = pc_4

# Apply initial transformation to point cloud:
source.transform(H_f4)

# Visazlize Raw transformed point clouds:
# o3d.visualization.draw_geometries([source, target, coordinate_frame])

# Preprocess point clouds:
voxel_size = 0.02  # means 5cm for this dataset
source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

# Execute global registration:
global_result = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
source.transform(global_result.transformation)

# Refine registration:
refined_result = refine_registration(source, target, source_fpfh, target_fpfh, global_result, voxel_size)
source.transform(refined_result.transformation)

# Display registration result:
draw_registration_result(source, target)
