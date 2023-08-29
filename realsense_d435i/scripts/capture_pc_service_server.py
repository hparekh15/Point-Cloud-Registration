#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Empty, EmptyResponse
import open3d as o3d

class PointCloudCapture:
    def __init__(self):
        self.pc2_msg = None
        pub = rospy.Publisher('point_cloud', PointCloud2, queue_size=10)
        sub = rospy.Subscriber('camera/depth/color/points', PointCloud2, self.callback_pc)
        capture_pc_service = rospy.Service('capture_pc_service', Empty, self.capture_pc)
        self.counter = 0

    def callback_pc(self, msg):
        self.pc2_msg = msg

    def capture_pc(self, req):
        
        rospy.loginfo("Capturing point cloud")
        if self.pc2_msg is None:
            rospy.logwarn("No PointCloud2 message received")
        else:
            np_pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.pc2_msg)
            print("PC shape", np_pc.shape)

            # Convert Numpy to Open3D PointCloud and process
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pc) # Convert Numpy to Open3D PointCloud
            processed_pc = self.process_pointcloud(pcd)

            # Convert back to numpy and then to PointCloud2 message and publish
            # np_points = np.asarray(processed_pc.points)
            # print("Processed PC shape", np_points.shape)
            # pc2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(np_points, self.pc2_msg.header)
            # self.pub.publish(pc2_msg) 
            # rospy.loginfo("Point cloud published")

            # Save point cloud to file locally
            o3d.io.write_point_cloud(f"pcd_{self.counter}.xyz", processed_pc)
            self.counter += 1
            

        return EmptyResponse()
    
    def process_pointcloud(self, o3d_pc):
        # Downsample point cloud
        o3d_pc = o3d_pc.voxel_down_sample(voxel_size=0.01)
        return o3d_pc

if __name__ == '__main__':
    rospy.init_node('capture_pc_service_server')
    PointCloudCapture = PointCloudCapture()
    
    rospy.spin()