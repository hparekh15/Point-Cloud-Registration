import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt


# Initialize coordinate frame at origin:
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0,0,0])

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

# Camera:
K = [602.94225,   0.     , 329.68172,
           0.     , 605.26279, 268.79021,
           0.     ,   0.     ,   1.     ]
cam_intrinsic = o3d.camera.PinholeCameraIntrinsic(width = int(640), height = int(480), fx = K[0], fy = K[4], cx = K[2], cy = K[5])
# cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])
cam2ee_pose = np.array([ 0.054997463118951415, -0.05209840989309977, 0.059835060796519936, -0.020268855234074387, -0.021384931834466995, 0.7000802683864398, 0.7134560084643664]) # New Calibration
H_ec = pose_to_trans_matrix(cam2ee_pose)

# Read poses:
pose_0 = np.array([0.5433591669998237, -0.0006044492074968256, 0.4461006797369755, 0.9988971854943165, -0.0019426293375725055, 0.045173900893987824, 0.012455696076060493])
pose_1 = np.array([0.28623804219160215, -0.3036154300992865, 0.2672033084755835, 0.8781077034229997, -0.28274842059731936, 0.0728706592009272, -0.37903132655759925])
pose_2 = np.array([0.6387597361618824, -0.13136177439893507, 0.24240477443755684, 0.8965175392246647, -0.3867020797207734, -0.15874663976277031, -0.1466683297010151])
pose_3 = np.array([0.6648028956400839, 0.10353065343650195,0.3181116637813364,0.9344161068713456,0.3405802385134069,-0.09899353598811891, 0.032656813117677604])
pose_4 = np.array([0.3102204189907309, 0.3012137491392316, 0.27004675879815443, 0.8121427953581528, 0.4004655602870154,0.030486245654598093,0.4232210062874868])

# Cal Transformation to robot base frame:
H_f0 = transformation_to_franka_base(pose_0, H_ec)
H_f1 = transformation_to_franka_base(pose_1, H_ec)
H_f2 = transformation_to_franka_base(pose_2, H_ec)
H_f3 = transformation_to_franka_base(pose_3, H_ec)
H_f4 = transformation_to_franka_base(pose_4, H_ec)

# Read depth img:
depth_0 = o3d.io.read_image("Test Data 4\depth_image_0.png")
depth_1 = o3d.io.read_image("Test Data 4\depth_image_1.png")
depth_2 = o3d.io.read_image("Test Data 4\depth_image_2.png")
depth_3 = o3d.io.read_image("Test Data 4\depth_image_3.png")
depth_4 = o3d.io.read_image("Test Data 4\depth_image_4.png")

# Read color img:
rgb_0 = o3d.io.read_image("Test Data 4\color_image_0.png")
rgb_1 = o3d.io.read_image("Test Data 4\color_image_1.png")
rgb_2 = o3d.io.read_image("Test Data 4\color_image_2.png")
rgb_3 = o3d.io.read_image("Test Data 4\color_image_3.png")
rgb_4 = o3d.io.read_image("Test Data 4\color_image_4.png")

# Create rgbd images:
rgbd_0 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_0, depth_0, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
rgbd_1 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_1, depth_1, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
rgbd_2 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_2, depth_2, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
rgbd_3 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_3, depth_3, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)
rgbd_4 = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_4, depth_4, depth_scale=1000.0, depth_trunc=2.0, convert_rgb_to_intensity=False)

# Plot images:
# plt.imshow(rgbd_4.depth)
# plt.imshow(rgbd_0.color)
# plt.show()

# Create point clouds from rgbd images:
pc_0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_0, cam_intrinsic)
pc_1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_1, cam_intrinsic)
pc_2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_2, cam_intrinsic)
pc_3 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_3, cam_intrinsic)
pc_4 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_4, cam_intrinsic)

# Transform oterwise pc will be upside down
pc_0.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pc_1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pc_2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pc_3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pc_4.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# pc_0.paint_uniform_color([1, 0, 0]) # red
# pc_1.paint_uniform_color([0, 1, 0]) # green
# pc_2.paint_uniform_color([0, 0, 1]) # blue
# pc_3.paint_uniform_color([1, 1, 0]) # yellow
# pc_4.paint_uniform_color([1, 0, 1]) # pink

# pc_0.transform(H_f0)
pc_1.transform(H_f1)

o3d.visualization.draw_geometries([pc_1, coordinate_frame])



