from pathlib import Path

import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"],
)

lidar_token = sample["data"]["LIDAR_TOP"]

lidar_path, _, _ = nusc.get_sample_data(lidar_token)

point_cloud = LidarPointCloud.from_file(lidar_path)

# Берём только XYZ.
points = point_cloud.points[:3, :].T.astype(np.float64)

print("LiDAR points:", points.shape)

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)

# Показываем облако.
o3d.visualization.draw_geometries(
    [cloud],
    window_name="nuScenes LiDAR",
)