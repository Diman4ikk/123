from pathlib import Path

import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import (
    points_in_box as nuscenes_points_in_box,
)

DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")


# Параметры обработки LiDAR.
GROUND_DISTANCE_THRESHOLD = 0.15
RANSAC_ITERATIONS = 1000

DBSCAN_EPS = 0.5
DBSCAN_MIN_POINTS = 10
MIN_CLUSTER_POINTS = 15

# Рабочая область.
ROI_X = (2.0, 50.0)
ROI_Y = (-25.0, 25.0)
ROI_Z = (-2.0, 4.0)


def load_data():
    """Загружает LiDAR и эталонные 3D-боксы одного sample."""
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

    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    point_cloud = LidarPointCloud.from_file(lidar_path)
    points = point_cloud.points[:3, :].T.astype(np.float64)

    return points, boxes, scene["name"]

def remove_ground(points: np.ndarray) -> np.ndarray:
    """Удаляет плоскость земли методом RANSAC."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    o3d.utility.random.seed(42)

    _, ground_indices = cloud.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=RANSAC_ITERATIONS,
    )

    non_ground = cloud.select_by_index(
        ground_indices,
        invert=True,
    )

    return np.asarray(non_ground.points)


def filter_roi(points: np.ndarray) -> np.ndarray:
    """Оставляет точки в заданной рабочей области."""
    mask = (
        (points[:, 0] >= ROI_X[0])
        & (points[:, 0] <= ROI_X[1])
        & (points[:, 1] >= ROI_Y[0])
        & (points[:, 1] <= ROI_Y[1])
        & (points[:, 2] >= ROI_Z[0])
        & (points[:, 2] <= ROI_Z[1])
    )

    return points[mask]


def cluster_points(points: np.ndarray) -> np.ndarray:
    """Выделяет кластеры DBSCAN по горизонтальным координатам X и Y."""
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
    ).copy()

    # Удаляем слишком маленькие кластеры из числа кандидатов.
    for label in np.unique(labels):
        if label < 0:
            continue

        if np.count_nonzero(labels == label) < MIN_CLUSTER_POINTS:
            labels[labels == label] = -1

    return labels


def points_inside_box(
    points: np.ndarray,
    box,
    ) -> np.ndarray:
    """Возвращает маску точек внутри ориентированного 3D-бокса."""
    return nuscenes_points_in_box(box, points.T)


def center_in_roi(center: np.ndarray) -> bool:
    """Проверяет, попадает ли центр эталонного объекта в ROI."""
    return (
        ROI_X[0] <= center[0] <= ROI_X[1]
        and ROI_Y[0] <= center[1] <= ROI_Y[1]
        and ROI_Z[0] <= center[2] <= ROI_Z[1]
    )


def evaluate(
    points: np.ndarray,
    labels: np.ndarray,
    boxes: list,
) -> None:
    """Сопоставляет кластеры с эталонными боксами один к одному."""

    cluster_ids = sorted(
        int(label)
        for label in np.unique(labels)
        if label >= 0
    )

    relevant_boxes = [
        box
        for box in boxes
        if center_in_roi(np.asarray(box.center))
    ]

    cluster_masks = {
        cluster_id: labels == cluster_id
        for cluster_id in cluster_ids
    }

    # Считаем пересечение каждой пары «кластер — GT».
    candidates = []

    for gt_index, box in enumerate(relevant_boxes):
        gt_mask = points_inside_box(points, box)
        gt_point_count = int(gt_mask.sum())

        if gt_point_count == 0:
            continue

        for cluster_id, cluster_mask in cluster_masks.items():
            cluster_size = int(cluster_mask.sum())
            intersection = int(np.count_nonzero(gt_mask & cluster_mask))

            if intersection == 0:
                continue

            union = cluster_size + gt_point_count - intersection
            point_iou = intersection / union if union else 0.0

            candidates.append(
                {
                    "gt_index": gt_index,
                    "cluster_id": cluster_id,
                    "intersection": intersection,
                    "point_iou": point_iou,
                    "precision": intersection / cluster_size,
                    "recall": intersection / gt_point_count,
                }
            )

    # Сначала рассматриваем пары с наибольшим Point IoU.
    candidates.sort(
        key=lambda item: item["point_iou"],
        reverse=True,
    )

    matched_gt = {}
    used_clusters = set()

    # Учебное правило сопоставления, не финальный порог качества.
    min_point_iou = 0.10

    for candidate in candidates:
        gt_index = candidate["gt_index"]
        cluster_id = candidate["cluster_id"]

        if candidate["point_iou"] < min_point_iou:
            continue

        if gt_index in matched_gt or cluster_id in used_clusters:
            continue

        matched_gt[gt_index] = candidate
        used_clusters.add(cluster_id)

    print("\nONE-TO-ONE EVALUATION")
    print("=" * 70)
    print("Ground-truth boxes in ROI:", len(relevant_boxes))
    print("Detected clusters:", len(cluster_ids))

    for gt_index, box in enumerate(relevant_boxes):
        print(f"\nGT #{gt_index + 1}: {box.name}")

        match = matched_gt.get(gt_index)

        if match is None:
            print("  Match: none")
            continue

        print(f"  Cluster: {match['cluster_id']}")
        print(f"  Overlapping points: {match['intersection']}")
        print(f"  Point precision: {match['precision']:.3f}")
        print(f"  Point recall: {match['recall']:.3f}")
        print(f"  Point IoU: {match['point_iou']:.3f}")

    true_matches = len(matched_gt)

    print("\nSUMMARY")
    print("-" * 70)
    print("Matched GT objects:", true_matches)
    print("Unmatched GT objects:", len(relevant_boxes) - true_matches)
    print("Unmatched clusters:", len(cluster_ids) - len(used_clusters))


def main() -> None:
    points, boxes, scene_name = load_data()

    print("Scene:", scene_name)
    print("Original LiDAR points:", len(points))
    print("Ground-truth boxes in sample:", len(boxes))

    non_ground = remove_ground(points)
    print("After ground removal:", len(non_ground))

    roi_points = filter_roi(non_ground)
    print("After ROI filter:", len(roi_points))

    labels = cluster_points(roi_points)

    print(
        "Clusters:",
        len(set(labels.tolist()) - {-1}),
    )
    print("Noise points:", int(np.sum(labels == -1)))

    evaluate(roi_points, labels, boxes)


if __name__ == "__main__":
    main()