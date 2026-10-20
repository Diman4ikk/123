from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

# Параметры RANSAC.
GROUND_DISTANCE_THRESHOLD = 0.15
RANSAC_N = 3
RANSAC_ITERATIONS = 1000

# Параметры DBSCAN.
DBSCAN_EPS = 0.5
DBSCAN_MIN_POINTS = 10
MIN_CLUSTER_POINTS = 15
# Рабочая область перед автомобилем.
ROI_X_MIN = 2.0
ROI_X_MAX = 50.0

ROI_Y_MIN = -25.0
ROI_Y_MAX = 25.0

ROI_Z_MIN = -2.0
ROI_Z_MAX = 4.0

def load_lidar_points() -> np.ndarray:
    """Загружает XYZ-координаты точек LIDAR_TOP."""
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

    return point_cloud.points[:3, :].T.astype(np.float64)


def remove_ground(points: np.ndarray) -> np.ndarray:
    """Удаляет плоскость земли с помощью RANSAC."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    _, ground_indices = cloud.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERATIONS,
    )

    object_cloud = cloud.select_by_index(
        ground_indices,
        invert=True,
    )

    return np.asarray(object_cloud.points)

def cluster_points(points: np.ndarray) -> np.ndarray:
    """Кластеризует точки методом DBSCAN в горизонтальной плоскости XY."""

    points_xy = np.column_stack(
        (
            points[:, 0],
            points[:, 1],
            np.zeros(len(points)),
        )
    )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xy)

    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=True,
        )
    )

    return labels


def visualize_clusters(
    points: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Показывает кластеры разными цветами."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    max_label = labels.max()

    if max_label < 0:
        print("Кластеры не найдены.")
        cloud.paint_uniform_color([0.5, 0.5, 0.5])

        o3d.visualization.draw_geometries(
            [cloud],
            window_name="DBSCAN - no clusters",
        )
        return

    # Создаём цвет для каждого кластера.
    colors = np.zeros((len(points), 3), dtype=np.float64)

    rng = np.random.default_rng(42)

    cluster_colors = rng.uniform(
        0.1,
        1.0,
        size=(max_label + 1, 3),
    )

    for label in range(max_label + 1):
        mask = labels == label

        cluster_size = int(mask.sum())

        if cluster_size < MIN_CLUSTER_POINTS:
            labels[mask] = -1
            continue

        colors[mask] = cluster_colors[label]

    # Шум делаем серым.
    noise_mask = labels == -1
    colors[noise_mask] = [0.3, 0.3, 0.3]

    cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries(
        [cloud],
        window_name="DBSCAN clusters",
    )


def print_cluster_statistics(
    points: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Печатает размеры найденных кластеров."""
    valid_labels = sorted(
        label
        for label in np.unique(labels)
        if label >= 0
    )

    print("\nCLUSTERS")
    print("--------")

    for label in valid_labels:
        cluster_points = points[labels == label]

        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)

        size_xyz = max_xyz - min_xyz

        print(
            f"Cluster {label}: "
            f"points={len(cluster_points)}, "
            f"size=("
            f"{size_xyz[0]:.2f}, "
            f"{size_xyz[1]:.2f}, "
            f"{size_xyz[2]:.2f}"
            f")"
        )
def filter_roi(points: np.ndarray) -> np.ndarray:
    """Оставляет точки в рабочей области вокруг дороги."""
    mask = (
        (points[:, 0] >= ROI_X_MIN)
        & (points[:, 0] <= ROI_X_MAX)
        & (points[:, 1] >= ROI_Y_MIN)
        & (points[:, 1] <= ROI_Y_MAX)
        & (points[:, 2] >= ROI_Z_MIN)
        & (points[:, 2] <= ROI_Z_MAX)
    )

    return points[mask]

def main() -> None:
    points = load_lidar_points()

    print("Total points:", len(points))

    object_points = remove_ground(points)

    print("Non-ground points:", len(object_points))

    roi_points = filter_roi(object_points)

    print("ROI points:", len(roi_points))

    labels = cluster_points(roi_points)

    cluster_count = len(
        set(labels) - {-1}
    )

    noise_count = int(np.sum(labels == -1))

    print("\nDBSCAN RESULT")
    print("-------------")
    print("Clusters:", cluster_count)
    print("Noise points:", noise_count)

    print_cluster_statistics(
        roi_points,
        labels,
    )

    visualize_clusters(
        roi_points,
        labels,
    )

if __name__ == "__main__":
    main()