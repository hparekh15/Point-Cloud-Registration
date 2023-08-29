#!/usr/bin/env python3
# license removed for brevity
import rospy
import numpy as np
import ros_numpy
import rosbag
import sensor_msgs

def callback(pointcloud_msg):  
  pointcloud_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pointcloud_msg, remove_nans=True)
  print("Point cloud shape: "+str(pointcloud_np.shape), end="\r")
    
def receive_message():
  rospy.init_node('pointcloud_sub_py', anonymous=True)
  rospy.Subscriber('/camera/depth/color/points', sensor_msgs.msg._PointCloud2.PointCloud2, callback)
  rospy.spin()
  
if __name__ == '__main__':
  receive_message()
  print()