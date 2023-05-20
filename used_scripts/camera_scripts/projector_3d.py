#!/usr/bin/env python3
import copy

import numpy as np
import open3d as o3d
from open3d.examples.benchmark.benchmark_ransac import preprocess_point_cloud, execute_global_registration


# Code copied from main depthai repo, depthai_helpers/projector_3d.py


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    pass


class PointCloudVisualizer():
    def __init__(self, intrinsic_matrix, width, height):
        self.R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        self.depth_map = None
        self.rgb = None
        self.pcl = o3d.geometry.PointCloud()

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(width,
                                                                         height,
                                                                         intrinsic_matrix[0][0],
                                                                         intrinsic_matrix[1][1],
                                                                         intrinsic_matrix[0][2],
                                                                         intrinsic_matrix[1][2])
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Point Cloud")
        self.vis.add_geometry(self.pcl)
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        self.vis.add_geometry(origin)
        view_control = self.vis.get_view_control()
        view_control.set_constant_z_far(1000)
        self.isstarted = False
        self.storedPcl = o3d.geometry.PointCloud()
        self.flag = False
        self.voxel_size = 0.005

    def rgbd_to_projection(self, depth_map, rgb, downsample = True, remove_noise = False):
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image(depth_map)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(rgb.shape) != 3), depth_trunc=20000, depth_scale=1000.0
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.pinhole_camera_intrinsic)
        
        # if downsample:
        #     pcd = pcd.voxel_down_sample(voxel_size=0.005)

        if remove_noise:
            pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

        # tmp_pcl = copy.deepcopy(self.storedPcl) + copy.deepcopy(pcd)
        self.pcl.points = pcd.points
        self.pcl.colors = pcd.colors
        self.flag = True

        self.pcl.rotate(self.R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
        return self.pcl

    def store_pcd(self):
        if len(self.storedPcl.points) == 0:
            self.storedPcl = copy.deepcopy(self.pcl)
            return None
        source_down, source_fpfh = preprocess_point_cloud(self.pcl, self.voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(self.storedPcl, self.voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    self.voxel_size)
        source_temp = copy.deepcopy(source_down)
        target_temp = copy.deepcopy(target_down)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(result_ransac.transformation)
        self.storedPcl = source_temp

    def visualize_pcd(self):
        self.vis.update_geometry(self.pcl)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close_window(self):
        self.vis.destroy_window()
