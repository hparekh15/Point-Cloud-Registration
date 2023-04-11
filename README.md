# Point-Cloud-Stitching
RA work for GPIS project to capture, stitch and update pointclouds

capture_pc.py captures a pointcloud when users hits enter

generate3D.py stictches point clouds together using ICP given the homography between the views as list of Rotation and Translation Matrices.

update3D.py updates the our stictched point cloud with the latest pc captured.
