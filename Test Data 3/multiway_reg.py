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

# Downsample point clouds:
voxel_size = 0.02
pc_0 = pc_0.voxel_down_sample(voxel_size=voxel_size)
pc_1 = pc_1.voxel_down_sample(voxel_size=voxel_size)
pc_2 = pc_2.voxel_down_sample(voxel_size=voxel_size)
pc_3 = pc_3.voxel_down_sample(voxel_size=voxel_size)
pc_4 = pc_4.voxel_down_sample(voxel_size=voxel_size)

o3d.visualization.draw_geometries([pc_0, pc_1, pc_2, pc_3, pc_4, coordinate_frame])

# EE Pose wrt Franka Base, input fromat (x, y, z, qx, qy, qz, qw):
pose_0 = np.array([0.4388440032543044, -0.0075462174011609465, 0.4500227169578591, 0.9984167281317976, 0.013601041358285134, 0.054512975713610266, -0.001603105833463662])
pose_1 = np.array([0.593014982208658, -0.01440953865760553, 0.47093945587862895, 0.9966198132937542, 0.023327084766205035, -0.06750266719144281, -0.0405384860821075])
pose_2 = np.array([0.32951222364764665, -0.013353048925014155, 0.46197756245136895, 0.991296372296191, 0.0471032671125117, 0.12287748099797435, -0.003014132994755392])
pose_3 = np.array([0.41916590634776996, 0.27148212380121334, 0.4872046957739613, 0.978732990427817, 0.0949678428991891, 0.0629274282139764, 0.1705817177364905])
pose_4 = np.array([0.4628885303619983, -0.30476017841252845, 0.4869179383359469, 0.9855881122065405, 0.012462907015585067, -0.00924378778291739, -0.1684354100116081])

# Camera Pose wrt EE:
# cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])
cam2ee_pose = np.array([ 0.054997463118951415, -0.05209840989309977, 0.059835060796519936, -0.020268855234074387, -0.021384931834466995, 0.7000802683864398, 0.7134560084643664]) # New Calibration

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
pc_0.transform(H_f0)

H_f1 = transformation_to_franka_base(pose_1, H_ec)
pc_1.transform(H_f1)

H_f2 = transformation_to_franka_base(pose_2, H_ec)
pc_2.transform(H_f2)

H_f3 = transformation_to_franka_base(pose_3, H_ec)
pc_3.transform(H_f3)

H_f4 = transformation_to_franka_base(pose_4, H_ec)
pc_4.transform(H_f4) 

# Visazlize point clouds:
pc_list = [pc_0, pc_1, pc_2, pc_3, pc_4]
print("PC LIST LEN", len(pc_list))
o3d.visualization.draw_geometries(pc_list + [coordinate_frame])
# o3d.visualization.draw_geometries([pc_0, pc_1, pc_2, pc_3, pc_4, coordinate_frame])

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph

print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pc_list,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)
    
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
print("Transform points and display")
for point_id in range(len(pc_list)):
    print(pose_graph.nodes[point_id].pose)
    pc_list[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(pc_list+ [coordinate_frame])