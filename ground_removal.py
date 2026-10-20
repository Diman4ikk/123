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

points = point_cloud.points[:3, :].T.astype(np.float64)

print("Total points:", len(points))

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)

# Ищем плоскость дороги.
plane_model, inlier_indices = cloud.segment_plane(
    distance_threshold=0.15,
    ransac_n=3,
    num_iterations=1000,
)

a, b, c, d = plane_model

print("\nGround plane:")
print(f"{a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

print("Ground points:", len(inlier_indices))

# Точки земли.
ground_cloud = cloud.select_by_index(inlier_indices)

# Все остальные точки.
object_cloud = cloud.select_by_index(
    inlier_indices,
    invert=True,
)

print("Non-ground points:", len(object_cloud.points))

print("\nOpening visualization...")

o3d.visualization.draw_geometries(
    [ground_cloud],
    window_name="Ground points",
)

o3d.visualization.draw_geometries(
    [object_cloud],
    window_name="Non-ground points",
)