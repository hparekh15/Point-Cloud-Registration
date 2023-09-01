#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty, EmptyResponse
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class PointCloudMerge:
    def __init__(self):
        self.pc2_msg = None
        self.workspace_pc = None
        self.ee_pose = None
        
        self.pub = rospy.Publisher('workspace_pc', PointCloud2, queue_size=10)
        sub1 = rospy.Subscriber('camera/depth/color/points', PointCloud2, self.callback_pc)
        sub2 = rospy.Subscriber('/franka_state_controller/ee_pose', Pose, self.callback_ee)
        merge_pc_service = rospy.Service('merge_pc_service', Empty, self.merge_pc)
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1)]

        self.cam2ee_pose = np.array([0.039201112671531854,  -0.035492796694330614, 0.07041605649874202, 0.013750894072379439,-0.005628437392997249, 0.7093345722195382,  0.7047153313635571])
        self.H_ec = self.pose_to_trans_matrix(self.cam2ee_pose)
        
        self.icp_threshold = 0.02
        self.downsample_voxel_size = 0.01
        self.counter = 1

    def callback_pc(self, msg):
        self.pc2_msg = msg

    def callback_ee(self, msg):
        self.ee_pose = np.array([msg.position.x, msg.position.y, msg.position.z, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])

    def capture_pc(self):
        if self.ee_pose is None:
            rospy.logwarn("No ee_pose received")
        if self.pc2_msg is None:
            rospy.logwarn("No PointCloud2 message received")
        else:
            rospy.loginfo("Capturing point cloud")
            np_pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.pc2_msg)
            print("Captured PC shape", np_pc.shape)

            # Convert Numpy to Open3D PointCloud and process
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pc) 
            processed_pc = self.process_pointcloud(pcd)
            
        return processed_pc
    
    def process_pointcloud(self, o3d_pc):
        # Downsample point cloud
        o3d_pc = o3d_pc.voxel_down_sample(voxel_size=self.downsample_voxel_size)
        # Statistical outlier removal
        clean_cloud, ind = o3d_pc.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.9)
        return clean_cloud
    
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
    
    def merge_pc(self, req):
        if self.workspace_pc is None:
            # Capture. Transform and save 1st point cloud
            self.workspace_pc = self.capture_pc()	
            H_fe = self.pose_to_trans_matrix(self.ee_pose)
            H_fc = H_fe @ self.H_ec
            self.workspace_pc.transform(H_fc)
            o3d.io.write_point_cloud("workspace_pcd_0.xyz", self.workspace_pc)
            rospy.loginfo("0th captured")

            # Publish workspace point cloud as PC2 message
            # header = Header()
            # header.stamp = rospy.Time.now()
            # header.frame_id = "franka_base"
            # points = np.asarray(self.workspace_pc.points)
            # workspace_pc2_msg = pc2.create_cloud(header, self.fields, points)
            # self.pub.publish(workspace_pc2_msg)
            # rospy.loginfo("Workspace point cloud published")
        else:
            # Capture new pc and calculate Transformation to franka base frame
            o3d_pc = self.capture_pc()
            H_fe = self.pose_to_trans_matrix(self.ee_pose)
            H_fc = H_fe @ self.H_ec

            # Transform pc to franka base frame
            reg_p2p = o3d.pipelines.registration.registration_icp(o3d_pc, self.workspace_pc, self.icp_threshold, H_fc, o3d.pipelines.registration.TransformationEstimationPointToPoint())
            o3d_pc.transform(reg_p2p.transformation)

            # Save current point cloud to file locally
            o3d.io.write_point_cloud(f"pcd_{self.counter}.xyz", o3d_pc)
            rospy.loginfo(f"pcd_{self.counter} saved locally")

            # Merge point cloud with workspace point cloud
            rospy.loginfo("Merging point clouds")
            self.workspace_pc += o3d_pc
            self.workspace_pc = self.process_pointcloud(self.workspace_pc)

            # Save point cloud to file locally
            o3d.io.write_point_cloud(f"workspace_pcd_{self.counter}.xyz", self.workspace_pc)
            rospy.loginfo(f"workspace_pcd_{self.counter} saved locally")
            self.counter += 1

            # Publish workspace point cloud as PC2 message
            # header = Header()
            # header.stamp = rospy.Time.now()
            # header.frame_id = "franka_base"
            # points = np.asarray(self.workspace_pc.points)
            # workspace_pc2_msg = pc2.create_cloud(header, self.fields, points)
            # self.pub.publish(workspace_pc2_msg)
            # rospy.loginfo("Workspace point cloud published")

        return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('merge_pc_service_server')
    PointCloudMerge = PointCloudMerge()
    rospy.spin()