import copy
import time

import open3d as o3d
from open3d.examples.benchmark.benchmark_ransac import preprocess_point_cloud, execute_global_registration
from open3d.examples.open3d_example import draw_registration_result
from os import listdir
from os.path import isfile, join

from open3d.examples.pipelines.multiway_registration import full_registration

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def load_point_clouds(voxel_size=0.0):
    pcds = []
    mypath = "./pcds_multiway"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for path in onlyfiles:
        pcd = o3d.io.read_point_cloud(f"./{mypath}/{path}")
        pcd.estimate_normals()
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcds.append(pcd_down)

    return pcds

voxel_size=0.01
pcds = load_point_clouds(voxel_size=voxel_size)

max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    pose_graph = full_registration(pcds,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcds[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
pcd_combined_down.translate([0,0,0,], relative=True)
# o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[0,0,0],
                                  up=[-0.0694, -0.9768, 0.2024])





