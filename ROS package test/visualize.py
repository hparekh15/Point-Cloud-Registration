import open3d as o3d

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0,0,0])
pc_0 = o3d.io.read_point_cloud("ROS package test/Test Data 2/workspace_pcd_6.xyz", format='xyz')
pc_1 = o3d.io.read_point_cloud("ROS package test/Test Data 2/workspace_pcd_5.xyz", format='xyz')
pc_2 = o3d.io.read_point_cloud("ROS package test/Test Data 2/workspace_pcd_8.xyz", format='xyz')

pc_0 = pc_0.paint_uniform_color([0, 0, 1])
pc_1 = pc_1.paint_uniform_color([1, 0, 0])
pc_2 = pc_2.paint_uniform_color([0, 1, 0])

o3d.visualization.draw_geometries([pc_2, coordinate_frame])