import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

n = 3 # number of perspectives
pcd_1 = o3d.io.read_point_cloud("/home/harshil/Desktop/Point cloud test/test data/point_cloud_1.pcd")
pcd_2 = o3d.io.read_point_cloud("/home/harshil/Desktop/Point cloud test/test data/point_cloud_2.pcd")
pcd_3 = o3d.io.read_point_cloud("/home/harshil/Desktop/Point cloud test/test data/point_cloud_3.pcd")

left = R.from_quat([0.925, 0.0, 0.0, 0.381])

mid = R.from_quat([1.0, 0.0, 0.0, 0.0])
right = R.from_quat([-0.925, 0.0, 0.0, 0.381])
r01 = (left.inv() * mid).as_matrix()
r12 = (mid.inv() * right).as_matrix()
# print(r01)
print(r12)
t01 = np.array([0.0, 0.3, 0.0])
t12 = np.array([0.0, -0.3, 0.0])


# Stack rotation and translation matrices to create a homogeneous transformation matrix
H01 = np.vstack((np.hstack((r01, t01.reshape(3, 1))), np.array([0, 0, 0, 1])))
H12 = np.vstack((np.hstack((r12, t12.reshape(3, 1))), np.array([0, 0, 0, 1])))

pcd_1.transform(H01)
