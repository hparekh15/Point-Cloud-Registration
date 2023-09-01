#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty, EmptyResponse
import open3d as o3d

class PointCloudCapture:
    def __init__(self):
        self.pc2_msg = None
        self.ee_pose = None
        # pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)
        sub1 = rospy.Subscriber('camera/depth/color/points', PointCloud2, self.callback_pc)
        sub2 = rospy.Subscriber('/franka_state_controller/ee_pose', Pose, self.callback_ee)
        capture_pc_service = rospy.Service('capture_pc_service', Empty, self.capture_pc)
        self.counter = 0
        self.downsample_voxel_size = 0.01
        self.cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])
        self.H_ec = self.pose_to_trans_matrix(self.cam2ee_pose)

    def callback_pc(self, msg):
        self.pc2_msg = msg

    def callback_ee(self, msg):
        self.ee_pose = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def pose_to_trans_matrix(self, pose):
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

    def capture_pc(self, req):
        if self.ee_pose is None:
            rospy.logwarn("No ee_pose received")
        if self.pc2_msg is None:
            rospy.logwarn("No PointCloud2 message received")
        else:
            rospy.loginfo("Capturing point cloud")
            # Convert PointCloud2 message to Numpy
            np_pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.pc2_msg)
            print("PC shape", np_pc.shape)

            # Convert Numpy to Open3D PointCloud and process
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pc) # Convert Numpy to Open3D PointCloud
            processed_pc = self.process_pointcloud(pcd)
            
            # Transform pc to franka base frame
            H_fe = self.pose_to_trans_matrix(self.ee_pose)
            H_fc = H_fe @ self.H_ec
            processed_pc.transform(H_fc)

            # Convert to PointCloud2 message and publish (to be implemented)
            

            # Save point cloud to file locally
            o3d.io.write_point_cloud(f"pcd_{self.counter}.xyz", processed_pc)
            rospy.loginfo(f"pcd_{self.counter} saved locally")
            self.counter += 1

        return EmptyResponse()
    
    def process_pointcloud(self, o3d_pc):
        # Downsample point cloud
        o3d_pc = o3d_pc.voxel_down_sample(voxel_size=self.downsample_voxel_size)
        # Statistical outlier removal
        clean_cloud, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.9)
        return clean_cloud

if __name__ == '__main__':
    rospy.init_node('capture_pc_service_server')
    PointCloudCapture = PointCloudCapture()
    rospy.spin()