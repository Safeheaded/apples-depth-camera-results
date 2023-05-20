import copy
import time

import open3d as o3d
from open3d.examples.benchmark.benchmark_ransac import preprocess_point_cloud, execute_global_registration
from open3d.examples.open3d_example import draw_registration_result
from os import listdir
from os.path import isfile, join

from open3d.examples.pipelines.multiway_registration import full_registration

path = './results_pcd/multiway_all_files.pcd'

pcd = o3d.io.read_point_cloud(path)
pcd.translate([0, 0, 0])
o3d.visualization.draw_geometries([pcd])





