from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
from pyquaternion import Quaternion
from sklearn.cluster import DBSCAN


# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")
VERSION = "v1.0-mini"
SCENE_NAME = "scene-0061"
OUTPUT_DIR = Path("outputs") / "candidate_sample"

# 1 = current LiDAR scan only.
NSWEEPS = 1

# Ground removal via RANSAC
GROUND_SEED_Z_MAX = -1.0
GROUND_SEED_MAX_RANGE = 30.0
GROUND_DISTANCE_THRESHOLD = 0.15
RANSAC_ITERATIONS = 1000
GROUND_HEIGHT_CUTOFF = 0.12

# Statistical Outlier Removal (SOR)
SOR_NB_NEIGHBORS = 12
SOR_STD_RATIO = 1.2

# Region of interest (LIDAR_TOP coordinates)
ROI_X = (-25.0, 25.0)
ROI_Y = (2.0, 50.0)
ROI_Z = (-2.0, 4.0)

# DBSCAN clustering parameters
DBSCAN_EPS = 0.50
DBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_POINTS = 6
MAX_CANDIDATE_LENGTH = 14.0
MAX_CANDIDATE_WIDTH = 5.0
MAX_CANDIDATE_HEIGHT = 4.0

# Roadside barrier modular length (splits fused barriers into individual blocks)
BARRIER_MAX_SEGMENT_LEN = 2.8
BARRIER_TARGET_LEN = 2.0

# Terrain filter parameters (rejects flat ground/slope remnants)
MIN_PLANARITY_RATIO = 0.03
TERRAIN_ANGLE_THRESH_DEG = 30.0

# Projection & Visualization
NEAR_PLANE_EPS = 1e-3
MIN_PROJECTED_AREA = 4.0
MIN_VISIBLE_RATIO = 0.3
BOX_THICKNESS = 2
DRAW_LABELS = False

CLUSTER_PALETTE = [
    (0, 255, 0),     # Bright Green
    (255, 191, 0),   # Cyan-blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 165, 255),   # Deep orange
    (255, 100, 100), # Light blue
    (100, 255, 100), # Light green
    (200, 200, 0),   # Cyan
    (128, 255, 0),   # Spring green
    (238, 130, 238), # Violet
]


# ============================================================
# DATA LOADING
# ============================================================
def load_sample(nusc: NuScenes) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the first keyframe from the selected scene."""
    scene = next((item for item in nusc.scene if item["name"] == SCENE_NAME), None)
    if scene is None:
        raise RuntimeError(f"Scene {SCENE_NAME!r} was not found in nuScenes.")
    sample = nusc.get("sample", scene["first_sample_token"])
    return scene, sample


def load_sensor_data(
    nusc: NuScenes,
    sample: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    """Load LiDAR points, camera image, intrinsics and sensor records."""
    lidar_record = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    camera_record = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    lidar_path = DATA_ROOT / lidar_record["filename"]
    camera_path = DATA_ROOT / camera_record["filename"]

    if not lidar_path.is_file():
        raise FileNotFoundError(f"LiDAR file not found: {lidar_path}")
    if not camera_path.is_file():
        raise FileNotFoundError(f"Camera image not found: {camera_path}")

    if NSWEEPS > 1:
        point_cloud, _ = LidarPointCloud.from_file_multisweep(
            nusc,
            sample,
            chan="LIDAR_TOP",
            ref_chan="LIDAR_TOP",
            nsweeps=NSWEEPS,
        )
    else:
        point_cloud = LidarPointCloud.from_file(str(lidar_path))

    points = point_cloud.points[:3, :].T.astype(np.float64)
    image = cv2.imread(str(camera_path))
    if image is None:
        raise RuntimeError(f"Could not read camera image: {camera_path}")

    camera_calibration = nusc.get(
        "calibrated_sensor", camera_record["calibrated_sensor_token"]
    )
    intrinsic = np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64)
    return points, image, intrinsic, lidar_record, camera_record


# ============================================================
# LIDAR PREPROCESSING & FILTERING
# ============================================================
def remove_ground(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit ground plane with RANSAC and keep points above road."""
    if len(points) == 0:
        return points.copy(), np.empty(0, dtype=np.float64), np.array([0, 0, 1.0]), 0.0

    ranges = np.linalg.norm(points[:, :2], axis=1)
    seed_mask = (points[:, 2] < GROUND_SEED_Z_MAX) & (ranges < GROUND_SEED_MAX_RANGE)
    seed_points = points[seed_mask]
    if len(seed_points) < 100:
        seed_points = points

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(seed_points)

    o3d.utility.random.seed(42)
    plane, inliers = cloud.segment_plane(
        distance_threshold=GROUND_DISTANCE_THRESHOLD,
        ransac_n=3,
        num_iterations=RANSAC_ITERATIONS,
        probability=1.0,
    )
    a, b, c, d = plane
    print(f"Ground plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 | inliers: {len(inliers)}")

    normal = np.asarray([a, b, c], dtype=np.float64)
    norm_len = float(np.linalg.norm(normal))
    if norm_len < 1e-12:
        raise RuntimeError("RANSAC returned an invalid ground plane.")
    normal_unit = normal / norm_len
    offset = float(d / norm_len)

    # Orient normal upward (+Z)
    if normal_unit[2] < 0:
        normal_unit = -normal_unit
        offset = -offset

    signed_dist = points @ normal_unit + offset
    keep = signed_dist > GROUND_HEIGHT_CUTOFF
    return points[keep].copy(), signed_dist[keep].copy(), normal_unit, offset


def remove_noise(
    points: np.ndarray,
    heights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Statistical Outlier Removal (SOR) to clean dust and mist."""
    xyz = points[:, :3]
    if len(xyz) < SOR_NB_NEIGHBORS + 1:
        return points, heights
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    _, ind = cloud.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )
    ind = np.asarray(ind)
    return points[ind].copy(), heights[ind].copy()


def filter_roi(
    points: np.ndarray,
    heights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep points inside the configured 3D region of interest."""
    if len(points) == 0:
        return points.copy(), heights.copy()

    mask = (
        (points[:, 0] >= ROI_X[0])
        & (points[:, 0] <= ROI_X[1])
        & (points[:, 1] >= ROI_Y[0])
        & (points[:, 1] <= ROI_Y[1])
        & (points[:, 2] >= ROI_Z[0])
        & (points[:, 2] <= ROI_Z[1])
    )
    return points[mask].copy(), heights[mask].copy()


def is_terrain_like(
    points_xyz: np.ndarray,
    ground_normal: np.ndarray | None,
    ratio_thresh: float = MIN_PLANARITY_RATIO,
    angle_thresh_deg: float = TERRAIN_ANGLE_THRESH_DEG,
) -> bool:
    """Reject clusters that are nearly 2D flat and parallel to the road plane."""
    if ground_normal is None or len(points_xyz) < 10:
        return False
    centered = points_xyz - points_xyz.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    ratio = eigvals[0] / eigvals[-1]
    if ratio >= ratio_thresh:
        return False
    local_normal = eigvecs[:, 0]
    cos_angle = np.clip(abs(np.dot(local_normal, ground_normal)), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    return bool(angle_deg < angle_thresh_deg)


# ============================================================
# CLUSTERING & GEOMETRIC SPLITTING
# ============================================================
def footprint(xyz: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Calculate XY center, physical length, width and yaw via minAreaRect."""
    xy = np.asarray(xyz[:, :2], dtype=np.float32)
    if len(xy) == 0:
        raise ValueError("Cannot calculate footprint for empty cluster.")

    rect = cv2.minAreaRect(xy)
    center_xy = np.asarray(rect[0], dtype=np.float64)
    corners = cv2.boxPoints(rect).astype(np.float64)
    edges = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edges, axis=1)

    longest_edge_index = int(np.argmax(edge_lengths))
    longest_edge = edges[longest_edge_index]
    length = float(edge_lengths[longest_edge_index])
    width = float(np.min(edge_lengths))
    yaw = float(np.arctan2(longest_edge[1], longest_edge[0]))
    yaw = float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)
    return center_xy, length, width, yaw


def split_long_cluster_if_barrier(
    cluster: np.ndarray,
    max_len: float = BARRIER_MAX_SEGMENT_LEN,
    target_len: float = BARRIER_TARGET_LEN,
) -> list[np.ndarray]:
    """Split elongated continuous barrier clusters into individual ~2.0m blocks."""
    xyz = cluster[:, :3]
    center_xy, length, width, yaw = footprint(xyz)
    if length > max_len and width < 0.85:
        c_x, c_y = center_xy
        dx = xyz[:, 0] - c_x
        dy = xyz[:, 1] - c_y
        cos_y = np.cos(-yaw)
        sin_y = np.sin(-yaw)
        u = dx * cos_y - dy * sin_y  # coordinate along longitudinal axis
        num_splits = int(np.ceil(length / target_len))
        u_min, u_max = float(u.min()), float(u.max())
        step = (u_max - u_min) / num_splits
        sub_clusters = []
        for i in range(num_splits):
            s_mask = (u >= u_min + i * step) & (
                u < u_min + (i + 1) * step + (0.01 if i == num_splits - 1 else 0.0)
            )
            if np.count_nonzero(s_mask) >= MIN_CLUSTER_POINTS:
                sub_clusters.append(cluster[s_mask].copy())
        if len(sub_clusters) > 1:
            return sub_clusters
    return [cluster]


def cluster_points(
    points: np.ndarray,
    heights: np.ndarray,
    ground_normal: np.ndarray,
) -> list[np.ndarray]:
    """Perform DBSCAN clustering, filter terrain artifacts, and split long barriers."""
    if len(points) == 0:
        return []

    data = np.column_stack([points, heights])
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(points)

    clusters: list[np.ndarray] = []
    for label in np.unique(labels):
        if label < 0:
            continue
        c = data[labels == label]
        if len(c) < MIN_CLUSTER_POINTS:
            continue
        if is_terrain_like(c[:, :3], ground_normal):
            continue

        center_xy, length, width, _ = footprint(c[:, :3])
        height = float(np.ptp(c[:, 2]))
        if (
            length > MAX_CANDIDATE_LENGTH
            or width > MAX_CANDIDATE_WIDTH
            or height > MAX_CANDIDATE_HEIGHT
        ):
            continue

        split_pieces = split_long_cluster_if_barrier(c)
        for piece in split_pieces:
            if len(piece) >= MIN_CLUSTER_POINTS:
                clusters.append(piece)

    return clusters


# ============================================================
# 3D BOUNDING BOX GENERATION (ROAD-ALIGNED)
# ============================================================
def build_candidate_box(
    cluster: np.ndarray,
    candidate_id: int,
    ground_normal: np.ndarray,
    ground_offset: float,
) -> tuple[Box, np.ndarray, np.ndarray, float, float, bool]:
    """Build a 3D box anchored to the road plane so it sits on the pavement."""
    xyz = cluster[:, :3]
    center_xy, length, width, yaw = footprint(xyz)

    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())

    # Calculate exact road elevation at cluster's (X, Y) center
    z_road = float(
        -(center_xy[0] * ground_normal[0] + center_xy[1] * ground_normal[1] + ground_offset)
        / ground_normal[2]
    )

    # Anchor to road plane if lowest point is close to ground (< 0.85m above road)
    if (z_min - z_road) < 0.85:
        z_bottom = z_road
    else:
        z_bottom = z_min

    height = max(z_max - z_bottom, 0.4)
    center_z = z_bottom + height / 2.0

    length = max(length, 0.2)
    width = max(width, 0.2)

    center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)
    dimensions_lwh = np.array([length, width, height], dtype=np.float64)

    box = Box(
        center=center,
        size=np.array([width, length, height], dtype=np.float64),
        orientation=Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw),
        name="lidar_candidate",
        token=str(candidate_id),
    )

    aspect_ratio = float(width / max(length, 1e-6))
    is_degenerate = bool(aspect_ratio < 0.15 or height < 0.25)

    return box, center, dimensions_lwh, yaw, aspect_ratio, is_degenerate


# ============================================================
# COORDINATE TRANSFORMS & PROJECTION (LIDAR -> CAMERA)
# ============================================================
def transform_points_to_camera(
    points_lidar: np.ndarray,
    nusc: NuScenes,
    lidar_record: dict[str, Any],
    camera_record: dict[str, Any],
) -> np.ndarray:
    """Transform points from LIDAR_TOP coords to CAM_FRONT coords (N, 3)."""
    cs_lidar = nusc.get("calibrated_sensor", lidar_record["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", lidar_record["ego_pose_token"])
    cs_cam = nusc.get("calibrated_sensor", camera_record["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", camera_record["ego_pose_token"])

    pts = points_lidar[:, :3].copy().T
    # 1. LiDAR -> Ego (lidar timestamp)
    pts = Quaternion(cs_lidar["rotation"]).rotation_matrix @ pts + np.array(cs_lidar["translation"])[:, None]
    # 2. Ego -> Global
    pts = Quaternion(pose_lidar["rotation"]).rotation_matrix @ pts + np.array(pose_lidar["translation"])[:, None]
    # 3. Global -> Ego (camera timestamp)
    pts = pts - np.array(pose_cam["translation"])[:, None]
    pts = Quaternion(pose_cam["rotation"]).rotation_matrix.T @ pts
    # 4. Ego -> Camera
    pts = pts - np.array(cs_cam["translation"])[:, None]
    pts = Quaternion(cs_cam["rotation"]).rotation_matrix.T @ pts
    return pts.T


def transform_box_to_camera(
    box: Box,
    nusc: NuScenes,
    lidar_record: dict[str, Any],
    camera_record: dict[str, Any],
) -> Box:
    """Transform a Box from current LiDAR coordinates into camera coordinates."""
    cs_lidar = nusc.get("calibrated_sensor", lidar_record["calibrated_sensor_token"])
    pose_lidar = nusc.get("ego_pose", lidar_record["ego_pose_token"])
    cs_cam = nusc.get("calibrated_sensor", camera_record["calibrated_sensor_token"])
    pose_cam = nusc.get("ego_pose", camera_record["ego_pose_token"])

    box.rotate(Quaternion(cs_lidar["rotation"]))
    box.translate(np.asarray(cs_lidar["translation"], dtype=np.float64))
    box.rotate(Quaternion(pose_lidar["rotation"]))
    box.translate(np.asarray(pose_lidar["translation"], dtype=np.float64))
    box.translate(-np.asarray(pose_cam["translation"], dtype=np.float64))
    box.rotate(Quaternion(pose_cam["rotation"]).inverse)
    box.translate(-np.asarray(cs_cam["translation"], dtype=np.float64))
    box.rotate(Quaternion(cs_cam["rotation"]).inverse)
    return box


def copy_box(box: Box) -> Box:
    return Box(
        center=box.center.copy(),
        size=box.wlh.copy(),
        orientation=Quaternion(box.orientation.elements.tolist()),
        name=box.name,
        token=box.token,
    )


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _camera_corners_clipped_to_near_plane(
    box: Box,
    near_eps: float = NEAR_PLANE_EPS,
) -> np.ndarray:
    corners = box.corners().T
    points: list[np.ndarray] = [p.copy() for p in corners if p[2] >= near_eps]
    for start, end in BOX_EDGES:
        p1 = corners[start]
        p2 = corners[end]
        z1, z2 = float(p1[2]), float(p2[2])
        if (z1 < near_eps and z2 > near_eps) or (z2 < near_eps and z1 > near_eps):
            denominator = z2 - z1
            if abs(denominator) < 1e-12:
                continue
            t = (near_eps - z1) / denominator
            intersection = p1 + t * (p2 - p1)
            intersection[2] = near_eps
            points.append(intersection)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def project_box_to_image(
    box: Box,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[tuple[int, int, int, int], float] | None:
    camera_points = _camera_corners_clipped_to_near_plane(box)
    if len(camera_points) < 3:
        return None

    projected = view_points(camera_points.T, intrinsic, normalize=True)
    if not np.all(np.isfinite(projected[:2])):
        return None

    x_min = float(projected[0].min())
    y_min = float(projected[1].min())
    x_max = float(projected[0].max())
    y_max = float(projected[1].max())
    if x_max < 0 or y_max < 0 or x_min >= image_width or y_min >= image_height:
        return None

    full_area = max((x_max - x_min) * (y_max - y_min), 1e-6)
    x1 = max(0, int(np.floor(x_min)))
    y1 = max(0, int(np.floor(y_min)))
    x2 = min(image_width - 1, int(np.ceil(x_max)))
    y2 = min(image_height - 1, int(np.ceil(y_max)))
    if x2 <= x1 or y2 <= y1:
        return None

    clipped_area = float((x2 - x1) * (y2 - y1))
    if clipped_area < MIN_PROJECTED_AREA:
        return None
    visible_ratio = min(clipped_area / full_area, 1.0)
    if visible_ratio < MIN_VISIBLE_RATIO:
        return None
    return (x1, y1, x2, y2), float(visible_ratio)


def _draw_projected_box_edges(
    image: np.ndarray,
    box: Box,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    corners = box.corners().T
    image_height, image_width = image.shape[:2]
    for start, end in BOX_EDGES:
        p1 = corners[start].copy()
        p2 = corners[end].copy()
        z1, z2 = float(p1[2]), float(p2[2])
        if z1 < NEAR_PLANE_EPS and z2 < NEAR_PLANE_EPS:
            continue
        if (z1 < NEAR_PLANE_EPS) != (z2 < NEAR_PLANE_EPS):
            denom = z2 - z1
            if abs(denom) < 1e-12:
                continue
            t = (NEAR_PLANE_EPS - z1) / denom
            intersection = p1 + t * (p2 - p1)
            intersection[2] = NEAR_PLANE_EPS
            if z1 < NEAR_PLANE_EPS:
                p1 = intersection
            else:
                p2 = intersection

        projected = view_points(np.column_stack((p1, p2)), intrinsic, normalize=True)
        if not np.all(np.isfinite(projected[:2])):
            continue
        a = tuple(np.round(projected[:2, 0]).astype(int))
        b = tuple(np.round(projected[:2, 1]).astype(int))
        visible, clipped_a, clipped_b = cv2.clipLine(
            (0, 0, image_width, image_height), a, b
        )
        if visible:
            cv2.line(image, clipped_a, clipped_b, color, thickness, cv2.LINE_AA)


# ============================================================
# MAIN PIPELINE
# ============================================================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "points").mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=VERSION, dataroot=str(DATA_ROOT), verbose=False)
    scene, sample = load_sample(nusc)
    points, image, intrinsic, lidar_record, camera_record = load_sensor_data(nusc, sample)
    image_height, image_width = image.shape[:2]

    # Save Ground Truth reference image
    _, ground_truth_boxes, _ = nusc.get_sample_data(
        camera_record["token"], box_vis_level=BoxVisibility.ANY
    )
    gt_image = image.copy()
    for gt_box in ground_truth_boxes:
        _draw_projected_box_edges(gt_image, gt_box, intrinsic, (0, 255, 0), 2)
    ground_truth_output = OUTPUT_DIR / f"{SCENE_NAME}_cam_front_ground_truth.jpg"
    if not cv2.imwrite(str(ground_truth_output), gt_image):
        raise RuntimeError(f"Could not save ground-truth visualization: {ground_truth_output}")

    print("=" * 60)
    print("LIDAR-ONLY CLUSTERING + CAMERA VISUALIZATION")
    print("=" * 60)
    print(f"Scene: {scene['name']} | Sample: {sample['token']}")
    print(f"Original LiDAR points: {len(points)}")

    np.save(OUTPUT_DIR / "raw_points.npy", points.astype(np.float32), allow_pickle=False)

    # 1. RANSAC Ground Removal
    non_ground, heights, ground_normal, ground_offset = remove_ground(points)
    np.save(OUTPUT_DIR / "non_ground_points.npy", non_ground.astype(np.float32), allow_pickle=False)
    print(f"After ground removal: {len(non_ground)}")

    # 2. SOR Noise Filter
    clean_points, clean_heights = remove_noise(non_ground, heights)
    print(f"After SOR noise removal: {len(clean_points)}")

    # 3. ROI Filter
    roi_points, roi_heights = filter_roi(clean_points, clean_heights)
    np.save(OUTPUT_DIR / "roi_points.npy", roi_points.astype(np.float32), allow_pickle=False)
    print(f"After ROI filter: {len(roi_points)}")

    # 4. Project ROI 3D LiDAR Points onto 2D Camera Image (Depth Overlay)
    pts_cam = transform_points_to_camera(roi_points, nusc, lidar_record, camera_record)
    depths = pts_cam[:, 2]
    in_front = depths > 1.0

    uv_proj = view_points(pts_cam[in_front].T, intrinsic, normalize=True)[:2].T
    in_img_mask = (
        (uv_proj[:, 0] >= 0) & (uv_proj[:, 0] < image_width) &
        (uv_proj[:, 1] >= 0) & (uv_proj[:, 1] < image_height)
    )

    valid_u = uv_proj[in_img_mask, 0]
    valid_v = uv_proj[in_img_mask, 1]
    valid_depths = depths[in_front][in_img_mask]

    print(f"LiDAR 3D points projected onto CAM_FRONT 2D image: {len(valid_u)}")

    # Render depth-colored 3D points on 2D image
    points_depth_image = image.copy()
    points_depth_output = OUTPUT_DIR / f"{SCENE_NAME}_cam_front_lidar_points.jpg"
    if len(valid_depths):
        d_norm = np.clip((valid_depths - 3.0) / (45.0 - 3.0), 0.0, 1.0)
        cmap = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        for i in range(len(valid_u)):
            px = min(image_width - 1, max(0, int(round(valid_u[i]))))
            py = min(image_height - 1, max(0, int(round(valid_v[i]))))
            pt_color = tuple(int(c) for c in cmap[i, 0])
            cv2.circle(points_depth_image, (px, py), radius=2, color=pt_color, thickness=-1)

    if not cv2.imwrite(str(points_depth_output), points_depth_image):
        raise RuntimeError(f"Could not save projected LiDAR image: {points_depth_output}")
    print(f"Saved 3D points on 2D image to: {points_depth_output.resolve()}")

    # 5. DBSCAN Clustering & Barrier Splitting
    clusters = cluster_points(roi_points, roi_heights, ground_normal)
    print(f"Extracted geometric clusters: {len(clusters)}")

    # 6. Build Candidate 3D Boxes & Project onto 2D Image
    annotated_image = image.copy()
    candidate_records: list[dict[str, Any]] = []
    projected_points_records: list[dict[str, Any]] = []
    projected_count = 0
    not_projected_count = 0

    for candidate_id, cluster in enumerate(clusters):
        cand_color = CLUSTER_PALETTE[candidate_id % len(CLUSTER_PALETTE)]

        # Save cluster points
        points_filename = f"candidate_{candidate_id:03d}.npy"
        np.save(OUTPUT_DIR / "points" / points_filename, cluster[:, :3].astype(np.float32), allow_pickle=False)

        # Build road-aligned candidate box
        candidate_box, center, dimensions, yaw, aspect_ratio, is_degenerate = build_candidate_box(
            cluster, candidate_id, ground_normal, ground_offset
        )

        # Transform box to camera coordinates
        camera_box = transform_box_to_camera(copy_box(candidate_box), nusc, lidar_record, camera_record)
        projection_result = project_box_to_image(camera_box, intrinsic, image_width, image_height)

        # Project points of this cluster onto 2D image
        cluster_cam = transform_points_to_camera(cluster[:, :3], nusc, lidar_record, camera_record)
        c_depths = cluster_cam[:, 2]
        c_front = c_depths > 1.0

        if np.any(c_front):
            c_uv = view_points(cluster_cam[c_front].T, intrinsic, normalize=True)[:2].T
            c_in_img = (
                (c_uv[:, 0] >= 0) & (c_uv[:, 0] < image_width) &
                (c_uv[:, 1] >= 0) & (c_uv[:, 1] < image_height)
            )
            for pt_i in range(len(c_uv)):
                if c_in_img[pt_i]:
                    u_px = min(image_width - 1, max(0, int(round(c_uv[pt_i, 0]))))
                    v_px = min(image_height - 1, max(0, int(round(c_uv[pt_i, 1]))))
                    # Draw cluster point on candidate image
                    cv2.circle(annotated_image, (u_px, v_px), radius=3, color=cand_color, thickness=-1)
                    # Extract RGB from camera image
                    b_val, g_val, r_val = image[v_px, u_px]
                    projected_points_records.append({
                        "u": u_px,
                        "v": v_px,
                        "depth": float(c_depths[c_front][pt_i]),
                        "x_lidar": float(cluster[c_front][pt_i, 0]),
                        "y_lidar": float(cluster[c_front][pt_i, 1]),
                        "z_lidar": float(cluster[c_front][pt_i, 2]),
                        "r": int(r_val),
                        "g": int(g_val),
                        "b": int(b_val),
                        "cluster_id": candidate_id,
                    })

        if projection_result is not None:
            bbox_camera_xyxy, visible_ratio = projection_result
            projected_count += 1
            _draw_projected_box_edges(annotated_image, camera_box, intrinsic, cand_color, BOX_THICKNESS)
            if DRAW_LABELS:
                x1, y1 = bbox_camera_xyxy[:2]
                cv2.putText(
                    annotated_image, f"C{candidate_id:02d}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cand_color, 2, cv2.LINE_AA,
                )
        else:
            bbox_camera_xyxy = None
            visible_ratio = None
            not_projected_count += 1

        candidate_records.append({
            "candidate_id": candidate_id,
            "source": "lidar_clustering_road_aligned",
            "point_count": int(len(cluster)),
            "points_file": f"points/{points_filename}",
            "center_lidar_xyz": center.tolist(),
            "dimensions_lidar_lwh": dimensions.tolist(),
            "yaw_lidar_rad": float(yaw),
            "aspect_ratio_wl": round(float(aspect_ratio), 4),
            "is_degenerate_geometry": bool(is_degenerate),
            "min_height_above_ground": float(cluster[:, 3].min()),
            "projected_to_camera": projection_result is not None,
            "visible_ratio": visible_ratio,
            "bbox_camera_xyxy": list(bbox_camera_xyxy) if bbox_camera_xyxy else None,
        })

    # Save candidates image
    candidates_image_output = OUTPUT_DIR / f"{SCENE_NAME}_cam_front_candidates.jpg"
    if not cv2.imwrite(str(candidates_image_output), annotated_image):
        raise RuntimeError(f"Could not save candidate visualization: {candidates_image_output}")

    # Save candidates JSON
    candidates_json_output = OUTPUT_DIR / f"{SCENE_NAME}_candidates.json"
    metadata = {
        "scene": scene["name"],
        "sample_token": sample["token"],
        "candidate_source": "lidar_dbscan_road_aligned",
        "ground_truth_used_for_candidates": False,
        "camera": "CAM_FRONT",
        "lidar": "LIDAR_TOP",
        "total_lidar_points": int(len(points)),
        "non_ground_points": int(len(non_ground)),
        "points_after_sor": int(len(clean_points)),
        "points_in_roi": int(len(roi_points)),
        "points_projected_in_camera": len(valid_u),
        "cluster_count": len(clusters),
        "projected_box_count": projected_count,
        "not_projected_count": not_projected_count,
        "parameters": {
            "nsweeps": NSWEEPS,
            "ground_seed_z_max": GROUND_SEED_Z_MAX,
            "ground_seed_max_range": GROUND_SEED_MAX_RANGE,
            "ground_distance_threshold": GROUND_DISTANCE_THRESHOLD,
            "ground_height_cutoff": GROUND_HEIGHT_CUTOFF,
            "ransac_iterations": RANSAC_ITERATIONS,
            "sor_nb_neighbors": SOR_NB_NEIGHBORS,
            "sor_std_ratio": SOR_STD_RATIO,
            "roi_x": list(ROI_X),
            "roi_y": list(ROI_Y),
            "roi_z": list(ROI_Z),
            "dbscan_eps": DBSCAN_EPS,
            "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
            "min_cluster_points": MIN_CLUSTER_POINTS,
            "max_candidate_length": MAX_CANDIDATE_LENGTH,
            "max_candidate_width": MAX_CANDIDATE_WIDTH,
            "max_candidate_height": MAX_CANDIDATE_HEIGHT,
            "barrier_max_segment_len": BARRIER_MAX_SEGMENT_LEN,
            "barrier_target_len": BARRIER_TARGET_LEN,
            "min_planarity_ratio": MIN_PLANARITY_RATIO,
            "terrain_angle_thresh_deg": TERRAIN_ANGLE_THRESH_DEG,
        },
        "candidates": candidate_records,
    }
    with candidates_json_output.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Save projected Camera + LiDAR points multimodal dataset
    proj_points_json = OUTPUT_DIR / f"{SCENE_NAME}_projected_points.json"
    with proj_points_json.open("w", encoding="utf-8") as f:
        json.dump(projected_points_records, f, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Clusters: {len(clusters)}")
    print(f"Projected onto camera: {projected_count}")
    print(f"3D Points on 2D image: {points_depth_output.resolve()}")
    print(f"Candidate clusters on 2D image: {candidates_image_output.resolve()}")
    print(f"Projected point/color records JSON: {proj_points_json.resolve()}")
    print(f"Candidates metadata JSON: {candidates_json_output.resolve()}")


if __name__ == "__main__":
    main()
