from pathlib import Path

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

# Берём тот же sample, который уже нашли.
scene = nusc.scene[0]
sample = nusc.get(
    "sample",
    scene["first_sample_token"],
)

# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

camera_token = sample["data"]["CAM_FRONT"]

camera_path, _, camera_intrinsic = nusc.get_sample_data(
    camera_token,
)

print("CAMERA")
print("path:", camera_path)

image = Image.open(camera_path)

print("size:", image.size)
print("camera intrinsic:")
print(np.array(camera_intrinsic))

# ---------------------------------------------------------
# LIDAR
# ---------------------------------------------------------

lidar_token = sample["data"]["LIDAR_TOP"]

lidar_path, _, _ = nusc.get_sample_data(
    lidar_token,
)

print("\nLIDAR")
print("path:", lidar_path)

point_cloud = LidarPointCloud.from_file(lidar_path)

print("points shape:", point_cloud.points.shape)
print("number of points:", point_cloud.points.shape[1])

# ---------------------------------------------------------
# TIME
# ---------------------------------------------------------

camera_data = nusc.get(
    "sample_data",
    camera_token,
)

lidar_data = nusc.get(
    "sample_data",
    lidar_token,
)

print("\nTIMESTAMPS")
print("camera:", camera_data["timestamp"])
print("lidar :", lidar_data["timestamp"])

print("\nDONE")