import copy
import time

import open3d as o3d
from open3d.examples.benchmark.benchmark_ransac import preprocess_point_cloud, execute_global_registration
from open3d.examples.open3d_example import draw_registration_result
from os import listdir
from os.path import isfile, join


flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def load_point_clouds(voxel_size=0.0):
    pcds = []
    mypath = "../data"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for path in onlyfiles:
        pcd = o3d.io.read_point_cloud(f"./{mypath}/{path}")
        # pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)

    return pcds

voxel_size=0.01
pcds = load_point_clouds(voxel_size=voxel_size)

combined = o3d.geometry.PointCloud()

for pcl in pcds:
    if len(combined.points) == 0:
        combined = pcl
        continue
    source_down, source_fpfh = preprocess_point_cloud(pcl, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(combined, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    distance_threshold = voxel_size * 0.4
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    source_temp = copy.deepcopy(source_down)
    target_temp = copy.deepcopy(target_down)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(result_icp.transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)

    combined = source_temp + target_temp


o3d.visualization.draw_geometries([combined])





