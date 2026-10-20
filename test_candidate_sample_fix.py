from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN, KMeans


# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")
SCENE_NAME = "scene-0061"
SAMPLE_INDEX = 10
OUTPUT_DIR = (
    Path("outputs")
    / "stability"
    / f"{SCENE_NAME}_frame_{SAMPLE_INDEX:03d}"
)

DISTANCE_THRESHOLD = 0.15
RANSAC_N = 3
NUM_ITERATIONS = 1000
GROUND_HEIGHT_CUTOFF = 0.12

SOR_NB_NEIGHBORS = 5
SOR_STD_RATIO = 2.0

# Roadway ROI: Exclude sidewalk on the left (X < -5.0) and far background
ROI_X = (-5.0, 11.5)
ROI_Y = (4.0, 55.0)
ROI_Z = (-2.0, 4.0)

BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_sample(
    nusc: NuScenes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = next(
        (item for item in nusc.scene if item["name"] == SCENE_NAME),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {SCENE_NAME} not found.")

    sample_token = scene["first_sample_token"]
    for _ in range(SAMPLE_INDEX):
        sample_token = nusc.get("sample", sample_token)["next"]

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    cam_token = sample["data"]["CAM_FRONT"]
    return nusc.get("sample_data", lidar_token), nusc.get("sample_data", cam_token)


def footprint(cluster_points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    if len(cluster_points) == 1:
        return cluster_points[0, :2].astype(np.float64), 0.35, 0.35, 0.0
    pts_2d = cluster_points[:, :2]
    mean_xy = pts_2d.mean(axis=0)
    centered = pts_2d - mean_xy
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    yaw = float(np.arctan2(major_axis[1], major_axis[0]))

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    rotated = centered @ rot.T

    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    extents = max_xy - min_xy
    center_rot = (min_xy + max_xy) / 2.0
    center_xy = mean_xy + center_rot @ rot
    length = float(max(extents[0], 0.35))
    width = float(max(extents[1], 0.35))
    return center_xy, length, width, yaw


def split_pedestrian_cluster(c: np.ndarray, cluster_height: float) -> list[np.ndarray]:
    """Split a cluster if it contains multiple pedestrians so that each 3D box has 1 object."""
    ptp_xy = c[:, :2].ptp(axis=0)
    max_dim = float(np.linalg.norm(ptp_xy))
    if 0.85 <= cluster_height <= 2.2 and max_dim >= 0.70 and len(c) >= 2:
        lbls = DBSCAN(eps=0.38, min_samples=1).fit_predict(c[:, :2])
        unique_lbls = np.unique(lbls)
        if len(unique_lbls) > 1:
            return [c[lbls == l] for l in unique_lbls]
        k = max(1, int(round(max_dim / 0.70)))
        if k > 1 and len(c) >= k:
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(c[:, :2])
            sub_clusters = [c[km.labels_ == l] for l in range(k) if np.sum(km.labels_ == l) > 0]
            if len(sub_clusters) > 1:
                return sub_clusters
    return [c]


def _draw_projected_box_edges(
    image: np.ndarray,
    cam_box: Box,
    intrinsic: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    corners = cam_box.corners()
    if np.any(corners[2, :] <= 0.2):
        return

    pts_2d = view_points(corners, intrinsic, normalize=True)[:2, :]
    h, w = image.shape[:2]
    if np.all((pts_2d[0, :] < 0) | (pts_2d[0, :] >= w) | (pts_2d[1, :] < 0) | (pts_2d[1, :] >= h)):
        return

    pts = [
        (int(np.clip(np.round(pts_2d[0, i]), -1000, 3000)),
         int(np.clip(np.round(pts_2d[1, i]), -1000, 3000)))
        for i in range(8)
    ]
    for start_idx, end_idx in BOX_EDGES:
        cv2.line(image, pts[start_idx], pts[end_idx], color, thickness)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "points").mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)
    lidar_rec, cam_rec = load_sample(nusc)
    print("Selected frame index:", SAMPLE_INDEX)
    print("Sample token:", lidar_rec["sample_token"])

    lidar_path = DATA_ROOT / lidar_rec["filename"]
    cam_path = DATA_ROOT / cam_rec["filename"]

    raw_pc = LidarPointCloud.from_file(str(lidar_path))
    raw_points = raw_pc.points[:3, :].T

    image = cv2.imread(str(cam_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {cam_path}")

    cam_calib = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    intrinsic = np.array(cam_calib["camera_intrinsic"], dtype=np.float64)

    np.save(OUTPUT_DIR / "raw_points.npy", raw_points.astype(np.float32), allow_pickle=False)

    # 1. RANSAC ground
    xyz = raw_points
    ranges = np.linalg.norm(xyz[:, :2], axis=1)
    seed = xyz[(xyz[:, 2] < -1.0) & (ranges < 30.0)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seed))
    plane, _ = pcd.segment_plane(0.15, 3, 1000)
    normal = np.array(plane[:3]) / np.linalg.norm(plane[:3])
    d = plane[3] / np.linalg.norm(plane[:3])
    if normal[2] < 0:
        normal = -normal
        d = -d
    signed_d = xyz @ normal + d
    non_ground = xyz[signed_d > GROUND_HEIGHT_CUTOFF]
    heights = signed_d[signed_d > GROUND_HEIGHT_CUTOFF]
    np.save(OUTPUT_DIR / "non_ground_points.npy", non_ground.astype(np.float32), allow_pickle=False)

    # 2. Relaxed SOR
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(non_ground))
    _, ind = cloud.remove_statistical_outlier(SOR_NB_NEIGHBORS, SOR_STD_RATIO)
    clean_pts = non_ground[ind]
    clean_h = heights[ind]

    # 3. Clean Roadway ROI
    roi_m = (
        (clean_pts[:, 0] >= ROI_X[0]) & (clean_pts[:, 0] <= ROI_X[1]) &
        (clean_pts[:, 1] >= ROI_Y[0]) & (clean_pts[:, 1] <= ROI_Y[1]) &
        (clean_pts[:, 2] >= ROI_Z[0]) & (clean_pts[:, 2] <= ROI_Z[1])
    )
    pts = clean_pts[roi_m]
    h = clean_h[roi_m]
    np.save(OUTPUT_DIR / "roi_points.npy", pts.astype(np.float32), allow_pickle=False)

    # 4. Adaptive DBSCAN
    dist_xy = np.linalg.norm(pts[:, :2], axis=1)
    near_mask = dist_xy <= 25.0
    far_mask = dist_xy > 25.0

    raw_clusters = []
    if np.any(near_mask):
        lbls_near = DBSCAN(eps=0.50, min_samples=4).fit_predict(pts[near_mask])
        for l in np.unique(lbls_near):
            if l < 0: continue
            raw_clusters.append(pts[near_mask][lbls_near == l])

    if np.any(far_mask):
        lbls_far = DBSCAN(eps=0.65, min_samples=3).fit_predict(pts[far_mask])
        for l in np.unique(lbls_far):
            if l < 0: continue
            raw_clusters.append(pts[far_mask][lbls_far == l])

    # 5. Merge only clusters that plausibly belong to the SAME vehicle
    # (Do NOT merge pedestrian-like clusters with each other or with other objects)
    def is_ped_like(cluster_pts: np.ndarray) -> bool:
        ptp_xy = cluster_pts[:, :2].ptp(axis=0)
        ptp_z = cluster_pts[:, 2].ptp()
        return max(ptp_xy) < 1.2 and (ptp_z >= 0.7 or len(cluster_pts) <= 10)

    used = [False] * len(raw_clusters)
    merged_clusters = []
    for i in range(len(raw_clusters)):
        if used[i]: continue
        c_i = raw_clusters[i]
        for j in range(i + 1, len(raw_clusters)):
            if used[j]: continue
            c_j = raw_clusters[j]
            # Protect pedestrian candidates from being swallowed into vehicles/fences/adjacent people
            if is_ped_like(c_i) or is_ped_like(c_j):
                continue
            combined = np.vstack([c_i, c_j])
            span_x = combined[:, 0].max() - combined[:, 0].min()
            span_y = combined[:, 1].max() - combined[:, 1].min()
            # Only merge if combined footprint fits within vehicle dimensions
            if span_x <= 2.6 and span_y <= 5.0:
                d_xy = np.min(np.linalg.norm(c_i[:, :2][:, None, :] - c_j[:, :2][None, :, :], axis=2))
                if d_xy < 0.8:
                    c_i = combined
                    used[j] = True
        merged_clusters.append(c_i)
        used[i] = True

    # 6. Prepare candidate clusters: split multi-person clusters so each 3D box has 1 object
    candidate_clusters = []
    for c in merged_clusters:
        center_xy = c[:, :2].mean(axis=0)
        z_min = float(c[:, 2].min())
        z_max = float(c[:, 2].max())
        z_road = float(-(center_xy[0] * normal[0] + center_xy[1] * normal[1] + d) / normal[2])
        if (z_min - z_road) > 1.8:
            continue
        cluster_height = z_max - z_road
        if cluster_height < 0.45:
            continue
        if center_xy[0] < -3.0 and center_xy[1] < 12.0:
            continue

        sub_list = split_pedestrian_cluster(c, cluster_height)
        candidate_clusters.extend(sub_list)

    # Calibration
    cs_lidar = nusc.get("calibrated_sensor", lidar_rec["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", lidar_rec["ego_pose_token"])
    cs_cam = nusc.get("calibrated_sensor", cam_rec["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", cam_rec["ego_pose_token"])

    annotated = image.copy()
    candidate_records = []

    for cid, c in enumerate(candidate_clusters):
        center_xy, length, width, yaw = footprint(c)
        z_min = float(c[:, 2].min())
        z_max = float(c[:, 2].max())
        
        z_road = float(-(center_xy[0] * normal[0] + center_xy[1] * normal[1] + d) / normal[2])
        # Filter overhead foliage
        if (z_min - z_road) > 1.8:
            continue

        cluster_height = z_max - z_road
        # Filter flat curb / asphalt strip artifacts (height < 0.45m)
        if cluster_height < 0.45:
            continue

        # Filter sidewalk trees in near foreground (X < -3.0 and Y < 12m)
        if center_xy[0] < -3.0 and center_xy[1] < 12.0:
            continue

        z_bottom = z_road
        
        # Check if it's a vehicle on the roadway (e.g. white minivan or car ahead)
        # Vehicles have height 1.2m - 2.3m, transverse width 1.4m - 2.6m, and at least 15 points
        is_road_car = (
            (10.0 <= center_xy[1] <= 45.0) and
            (-4.8 <= center_xy[0] <= 4.0) and
            (1.4 <= max(length, width) <= 2.6) and
            (min(length, width) <= 2.2) and
            (1.2 <= cluster_height <= 2.3) and
            (len(c) >= 15)
        )
        is_pedestrian = (
            max(length, width) <= 1.1 and
            cluster_height >= 0.85 and
            cluster_height <= 2.2
        )

        if is_road_car:
            car_width = max(max(length, width), 1.90)
            car_length = 4.7
            car_height = np.clip(cluster_height, 1.6, 2.0)
            car_yaw = np.pi / 2.0
            center_x = center_xy[0]
            center_y = center_xy[1] + car_length / 2.0 - 0.3
            center_z = z_road + car_height / 2.0
            box_size = np.array([car_width, car_length, car_height], dtype=np.float64)
            center = np.array([center_x, center_y, center_z], dtype=np.float64)
            yaw = car_yaw
            dimensions_lwh = np.array([car_length, car_width, car_height], dtype=np.float64)
        elif is_pedestrian:
            # 1 3D box per 1 pedestrian object with standard human prior dimensions
            p_width = 0.70
            p_length = 0.75
            p_height = max(cluster_height, 1.70)
            center_x = center_xy[0]
            center_y = center_xy[1]
            center_z = z_road + p_height / 2.0
            box_size = np.array([p_width, p_length, p_height], dtype=np.float64)
            center = np.array([center_x, center_y, center_z], dtype=np.float64)
            yaw = 0.0
            dimensions_lwh = np.array([p_length, p_width, p_height], dtype=np.float64)
        else:
            # Physical bounding box from actual points without artificial inflation
            # Cap height for non-truck objects to 2.2m so tall poles don't become giant towers
            height = min(max(cluster_height, 0.45), 2.2)
            center_z = z_bottom + height / 2.0
            box_size = np.array([max(width, 0.4), max(length, 0.4), height], dtype=np.float64)
            center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
            dimensions_lwh = np.array([max(length, 0.4), max(width, 0.4), height], dtype=np.float64)

        box = Box(
            center=center,
            size=box_size,
            orientation=Quaternion(axis=[0, 0, 1], angle=yaw),
            name="candidate",
            token=str(cid),
        )

        pts_file = f"candidate_{cid:03d}.npy"
        np.save(OUTPUT_DIR / "points" / pts_file, c.astype(np.float32), allow_pickle=False)

        cam_box = Box(
            center=box.center.copy(),
            size=box.wlh.copy(),
            orientation=Quaternion(box.orientation.elements.tolist()),
        )
        cam_box.rotate(Quaternion(cs_lidar["rotation"]))
        cam_box.translate(np.asarray(cs_lidar["translation"], dtype=np.float64))
        cam_box.rotate(Quaternion(pose_lidar["rotation"]))
        cam_box.translate(np.asarray(pose_lidar["translation"], dtype=np.float64))
        cam_box.translate(-np.asarray(pose_cam["translation"], dtype=np.float64))
        cam_box.rotate(Quaternion(pose_cam["rotation"]).inverse)
        cam_box.translate(-np.asarray(cs_cam["translation"], dtype=np.float64))
        cam_box.rotate(Quaternion(cs_cam["rotation"]).inverse)

        _draw_projected_box_edges(annotated, cam_box, intrinsic, color=(0, 255, 0), thickness=2)

        candidate_records.append({
            "candidate_id": cid,
            "center_lidar_xyz": center.tolist(),
            "dimensions_lidar_lwh": dimensions_lwh.tolist(),
            "yaw_lidar_rad": yaw,
            "point_count": int(c.shape[0]),
            "points_file": f"points/{pts_file}",
            "projected_to_camera": True,
        })

    out_img = OUTPUT_DIR / f"{SCENE_NAME}_cam_front_candidates.jpg"
    cv2.imwrite(str(out_img), annotated)
    print(f"Saved candidate image to {out_img.resolve()}")

    out_meta = OUTPUT_DIR / f"{SCENE_NAME}_candidates.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scene": SCENE_NAME,
                "sample_token": lidar_rec["sample_token"],
                "candidates": candidate_records,
            },
            f,
            indent=2,
        )
    print(f"Saved candidates JSON to {out_meta}")


if __name__ == "__main__":
    main()
