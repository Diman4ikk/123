from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, points_in_box
from pyquaternion import Quaternion


# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")
VERSION = "v1.0-mini"
SCENE_NAME = "scene-0061"
SAMPLE_INDEX = 10
OUTPUT_DIR = (
    Path("outputs")
    / "stability"
    / f"{SCENE_NAME}_frame_{SAMPLE_INDEX:03d}"
)
CANDIDATES_PATH = OUTPUT_DIR / f"{SCENE_NAME}_candidates.json"
RAW_POINTS_PATH = OUTPUT_DIR / "raw_points.npy"
NON_GROUND_POINTS_PATH = OUTPUT_DIR / "non_ground_points.npy"
ROI_POINTS_PATH = OUTPUT_DIR / "roi_points.npy"
OUTPUT_PATH = OUTPUT_DIR / f"{SCENE_NAME}_evaluation.json"

# Must match the ROI used by prepare_candidate_sample.py.
ROI_X = (-5.5, 11.5)
ROI_Y = (4.0, 55.0)
ROI_Z = (-2.0, 4.0)

# Point-overlap diagnostic thresholds.
MIN_INTERSECTION_POINTS = 3
MIN_CANDIDATE_PURITY = 0.5

# Geometry evaluation: candidate and GT boxes are matched one-to-one
# independently at each 3D IoU threshold.
BOX_IOU_THRESHOLDS = (0.10, 0.25, 0.50)
DEFAULT_SAMPLE_TOKEN = "ca9a282c9e77460f8360f564131a8af5"


# ============================================================
# CATEGORY MAPPING
# ============================================================
def category_to_label(category: str) -> str | None:
    """Map nuScenes categories to the target labels used in this evaluation."""
    if category.startswith("vehicle.car"):
        return "car"
    if category.startswith("vehicle.truck"):
        return "truck"
    if category.startswith("vehicle.bus"):
        return "bus"
    if category.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category.startswith("vehicle.bicycle"):
        return "bicycle"
    if category.startswith("vehicle.construction"):
        return "special_machinery"
    if category.startswith("human.pedestrian."):
        return "pedestrian"
    if category == "movable_object.barrier":
        return "barrier"
    if category == "movable_object.trafficcone":
        return "traffic_cone"
    return None


# ============================================================
# LOADING
# ============================================================
def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)
    return data


def load_points(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Point file not found: {path}\n"
            "Run prepare_candidate_sample.py first to regenerate the point files."
        )
    points = np.load(path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an Nx3 points array, got {points.shape}: {path}")
    return np.asarray(points[:, :3], dtype=np.float32)


def load_candidates() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_json(CANDIDATES_PATH)
    if data.get("ground_truth_used_for_candidates", False):
        raise ValueError(
            "Candidate generation used ground truth; this evaluation would not be independent."
        )
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("JSON does not contain a valid 'candidates' list.")
    return data, candidates


def get_sample_token(data: dict[str, Any]) -> str:
    token = data.get("sample_token") or data.get("metadata", {}).get("sample_token")
    if not token:
        token = DEFAULT_SAMPLE_TOKEN
    if not isinstance(token, str):
        raise ValueError("Invalid sample_token in candidates JSON.")
    return token


# ============================================================
# GROUND TRUTH
# ============================================================
def intersects_roi(box: Box) -> bool:
    """Return True if the GT box's AABB intersects the configured 3D ROI."""
    corners = box.corners()
    min_x, min_y, min_z = corners.min(axis=1)
    max_x, max_y, max_z = corners.max(axis=1)
    return bool(
        max_x >= ROI_X[0]
        and min_x <= ROI_X[1]
        and max_y >= ROI_Y[0]
        and min_y <= ROI_Y[1]
        and max_z >= ROI_Z[0]
        and min_z <= ROI_Z[1]
    )


def load_ground_truth(nusc: NuScenes, sample_token: str) -> list[dict[str, Any]]:
    """Load target GT boxes in the current LIDAR_TOP coordinate frame."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    camera_token = sample["data"]["CAM_FRONT"]

    _, lidar_boxes, _ = nusc.get_sample_data(
        lidar_token, box_vis_level=BoxVisibility.NONE
    )
    _, camera_boxes, _ = nusc.get_sample_data(
        camera_token, box_vis_level=BoxVisibility.ANY
    )
    visible_tokens = {
        box.token for box in camera_boxes if category_to_label(box.name) is not None
    }

    ground_truth: list[dict[str, Any]] = []
    for box in lidar_boxes:
        label = category_to_label(box.name)
        if label is None or box.token not in visible_tokens:
            continue
        if not intersects_roi(box):
            continue
        ground_truth.append(
            {
                "token": box.token,
                "category": box.name,
                "label": label,
                "box": box,
            }
        )
    return ground_truth


# ============================================================
# CANDIDATE HELPERS
# ============================================================
def candidate_id(candidate: dict[str, Any], index: int) -> str:
    value = candidate.get("candidate_id", candidate.get("id", index))
    if isinstance(value, int):
        return f"C{value:02d}"
    text = str(value)
    if text.isdigit():
        return f"C{int(text):02d}"
    return text


def candidate_points_path(candidate: dict[str, Any]) -> Path:
    points_file = candidate.get("points_file")
    if not isinstance(points_file, str) or not points_file:
        raise ValueError(
            f"Candidate {candidate.get('candidate_id', '?')} has no 'points_file'. "
            "Regenerate candidates with prepare_candidate_sample.py."
        )
    path = Path(points_file)
    if not path.is_absolute():
        path = CANDIDATES_PATH.parent / path
    return path


def is_projected(candidate: dict[str, Any]) -> bool:
    value = candidate.get("projected_to_camera")
    return value if isinstance(value, bool) else False


def candidate_to_box(candidate: dict[str, Any], index: int) -> Box:
    """Rebuild a candidate Box from geometry saved by prepare_candidate_sample.py."""
    center = candidate.get("center_lidar_xyz")
    dimensions = candidate.get("dimensions_lidar_lwh")
    yaw = candidate.get("yaw_lidar_rad")

    if not isinstance(center, list) or len(center) != 3:
        raise ValueError(
            f"Candidate {candidate_id(candidate, index)} has invalid center_lidar_xyz."
        )
    if not isinstance(dimensions, list) or len(dimensions) != 3:
        raise ValueError(
            f"Candidate {candidate_id(candidate, index)} has invalid dimensions_lidar_lwh."
        )
    if not isinstance(yaw, (int, float)):
        raise ValueError(
            f"Candidate {candidate_id(candidate, index)} has invalid yaw_lidar_rad."
        )

    length, width, height = (float(value) for value in dimensions)
    if min(length, width, height) <= 0:
        raise ValueError(
            f"Candidate {candidate_id(candidate, index)} has non-positive box dimensions."
        )

    return Box(
        center=np.asarray(center, dtype=np.float64),
        size=np.asarray([width, length, height], dtype=np.float64),
        orientation=Quaternion(axis=[0.0, 0.0, 1.0], radians=float(yaw)),
        name="lidar_candidate",
        token=str(candidate.get("candidate_id", index)),
    )


def count_points_inside(box: Box, points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    return int(np.count_nonzero(points_in_box(box, points.T)))


# ============================================================
# 3D ORIENTED BOX IOU AND AXIS DIAGNOSTICS
# ============================================================
def box_iou_3d(box_a: Box, box_b: Box) -> float:
    """Calculate oriented 3D IoU from XY footprint intersection and Z overlap."""
    corners_a = box_a.corners()
    corners_b = box_b.corners()
    polygon_a = np.ascontiguousarray(
        box_a.bottom_corners()[:2, :].T, dtype=np.float32
    )
    polygon_b = np.ascontiguousarray(
        box_b.bottom_corners()[:2, :].T, dtype=np.float32
    )

    try:
        intersection_area, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    except cv2.error:
        return 0.0

    area_a = float(box_a.wlh[0] * box_a.wlh[1])
    area_b = float(box_b.wlh[0] * box_b.wlh[1])
    intersection_area = float(
        np.clip(float(intersection_area), 0.0, min(area_a, area_b))
    )

    z_a_min, z_a_max = float(corners_a[2, :].min()), float(corners_a[2, :].max())
    z_b_min, z_b_max = float(corners_b[2, :].min()), float(corners_b[2, :].max())
    intersection_height = max(0.0, min(z_a_max, z_b_max) - max(z_a_min, z_b_min))
    intersection_volume = intersection_area * intersection_height
    volume_a = float(np.prod(box_a.wlh))
    volume_b = float(np.prod(box_b.wlh))
    union_volume = volume_a + volume_b - intersection_volume
    if union_volume <= 0.0:
        return 0.0
    return float(np.clip(intersection_volume / union_volume, 0.0, 1.0))


def box_footprint_major_axis_angle(box: Box) -> float:
    """Return the physical major-axis angle in [0, pi), independent of WLH labels."""
    corners = box.bottom_corners()[:2, :].T
    edges = np.roll(corners, -1, axis=0) - corners
    edge_lengths = np.linalg.norm(edges, axis=1)
    longest_idx = int(np.argmax(edge_lengths))
    edge = edges[longest_idx]
    angle = float(np.arctan2(edge[1], edge[0]))
    return angle % np.pi


def axis_difference_radians(angle_a: float, angle_b: float) -> float:
    """Smallest signed difference between undirected axes (180-degree symmetry)."""
    return float((angle_a - angle_b + np.pi / 2.0) % np.pi - np.pi / 2.0)


def maximum_cardinality_iou_matching(
    pair_scores: list[dict[str, Any]],
    candidate_count: int,
    gt_count: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Find maximum-cardinality one-to-one matches among pairs above an IoU threshold.

    A greedy highest-IoU assignment can leave a GT unmatched even when another
    valid one-to-one assignment would match more GT objects. This augmenting-path
    matcher maximizes match count; candidate adjacency is ordered by IoU to make
    tie choices deterministic and favor higher-IoU alternatives where possible.
    """
    adjacency: list[list[tuple[int, float, dict[str, Any]]]] = [
        [] for _ in range(candidate_count)
    ]
    for pair in pair_scores:
        score = float(pair["iou_3d"])
        if score >= threshold:
            adjacency[int(pair["candidate_index"])].append(
                (int(pair["gt_index"]), score, pair)
            )
    for neighbors in adjacency:
        neighbors.sort(key=lambda item: item[1], reverse=True)

    candidate_order = sorted(
        range(candidate_count),
        key=lambda idx: max((item[1] for item in adjacency[idx]), default=0.0),
        reverse=True,
    )
    gt_to_candidate: list[int | None] = [None] * gt_count
    gt_to_pair: list[dict[str, Any] | None] = [None] * gt_count

    def augment(candidate_idx: int, seen_gt: set[int]) -> bool:
        for gt_idx, _score, pair in adjacency[candidate_idx]:
            if gt_idx in seen_gt:
                continue
            seen_gt.add(gt_idx)
            previous_candidate = gt_to_candidate[gt_idx]
            if previous_candidate is None or augment(previous_candidate, seen_gt):
                gt_to_candidate[gt_idx] = candidate_idx
                gt_to_pair[gt_idx] = pair
                return True
        return False

    for candidate_idx in candidate_order:
        if adjacency[candidate_idx]:
            augment(candidate_idx, set())

    matches = [pair for pair in gt_to_pair if pair is not None]
    matches.sort(key=lambda pair: float(pair["iou_3d"]), reverse=True)
    return matches


def evaluate_boxes_by_iou(
    candidates: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate 3D geometry with maximum-cardinality matching at each IoU threshold."""
    candidate_boxes = [
        candidate_to_box(candidate, index) for index, candidate in enumerate(candidates)
    ]
    pair_scores: list[dict[str, Any]] = []
    best_per_candidate: list[dict[str, Any]] = []
    best_iou_per_gt = [0.0] * len(ground_truth)

    for candidate_index, (candidate, candidate_box) in enumerate(
        zip(candidates, candidate_boxes, strict=True)
    ):
        cid = candidate_id(candidate, candidate_index)
        best_match: dict[str, Any] | None = None

        for gt_index, gt in enumerate(ground_truth):
            gt_box = gt["box"]
            iou = box_iou_3d(candidate_box, gt_box)
            best_iou_per_gt[gt_index] = max(best_iou_per_gt[gt_index], iou)
            pair_scores.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_id": cid,
                    "gt_index": gt_index,
                    "gt_token": gt["token"],
                    "gt_label": gt["label"],
                    "iou_3d": float(iou),
                }
            )

            if best_match is None or iou > best_match["iou_3d"]:
                candidate_major = box_footprint_major_axis_angle(candidate_box)
                gt_major = box_footprint_major_axis_angle(gt_box)
                major_axis_difference = axis_difference_radians(candidate_major, gt_major)
                best_match = {
                    "gt_token": gt["token"],
                    "gt_label": gt["label"],
                    "gt_category": gt["category"],
                    "iou_3d": float(iou),
                    "candidate_center": [float(value) for value in candidate_box.center],
                    "gt_center": [float(value) for value in gt_box.center],
                    "center_distance_m": float(
                        np.linalg.norm(candidate_box.center - gt_box.center)
                    ),
                    "candidate_dimensions_lwh": [
                        float(candidate_box.wlh[1]),
                        float(candidate_box.wlh[0]),
                        float(candidate_box.wlh[2]),
                    ],
                    "gt_dimensions_lwh": [
                        float(gt_box.wlh[1]),
                        float(gt_box.wlh[0]),
                        float(gt_box.wlh[2]),
                    ],
                    # These are directions of physical long edges, not raw Box yaw values.
                    "candidate_major_axis_deg": float(np.degrees(candidate_major)),
                    "gt_major_axis_deg": float(np.degrees(gt_major)),
                    "major_axis_difference_deg": float(abs(np.degrees(major_axis_difference))),
                    # Backward-compatible alias for older report readers.
                    "yaw_difference_deg": float(abs(np.degrees(major_axis_difference))),
                }

        if best_match is not None and best_match["iou_3d"] <= 1e-9:
            best_match = None
        best_per_candidate.append(
            {
                "candidate_id": cid,
                "projected_to_camera": is_projected(candidate),
                "is_degenerate_geometry": bool(candidate.get("is_degenerate_geometry", False)),
                "best_match": best_match,
            }
        )

    threshold_results: dict[str, Any] = {}
    by_threshold_per_gt: dict[str, set[str]] = {}
    for threshold in BOX_IOU_THRESHOLDS:
        matches = maximum_cardinality_iou_matching(
            pair_scores,
            candidate_count=len(candidates),
            gt_count=len(ground_truth),
            threshold=threshold,
        )
        used_candidates = {int(pair["candidate_index"]) for pair in matches}
        used_gt = {int(pair["gt_index"]) for pair in matches}
        detected_tokens = {ground_truth[idx]["token"] for idx in used_gt}

        by_class: dict[str, dict[str, int]] = {}
        for gt_index, gt in enumerate(ground_truth):
            label = gt["label"]
            by_class.setdefault(label, {"total": 0, "detected": 0})
            by_class[label]["total"] += 1
            if gt_index in used_gt:
                by_class[label]["detected"] += 1

        key = f"{threshold:.2f}"
        by_threshold_per_gt[key] = detected_tokens
        threshold_results[key] = {
            "iou_threshold": threshold,
            "matching_algorithm": "maximum_cardinality_bipartite_matching",
            "matched_gt_objects": len(used_gt),
            "total_gt_objects": len(ground_truth),
            "unmatched_gt_objects": len(ground_truth) - len(used_gt),
            "recall": len(used_gt) / len(ground_truth) if ground_truth else 0.0,
            "matched_candidates": len(used_candidates),
            "matches": [
                {
                    "candidate_id": pair["candidate_id"],
                    "gt_token": pair["gt_token"],
                    "gt_label": pair["gt_label"],
                    "iou_3d": float(pair["iou_3d"]),
                }
                for pair in matches
            ],
            "by_class": by_class,
        }

    gt_best_results = []
    for index, gt in enumerate(ground_truth):
        gt_best_results.append(
            {
                "gt_token": gt["token"],
                "label": gt["label"],
                "category": gt["category"],
                "best_candidate_iou_3d": float(best_iou_per_gt[index]),
                "matched_at_thresholds": {
                    threshold: gt["token"] in tokens
                    for threshold, tokens in by_threshold_per_gt.items()
                },
            }
        )

    return {
        "description": (
            "Oriented 3D box IoU from rotated XY footprint intersection and vertical overlap. "
            "Recall matching uses maximum-cardinality one-to-one bipartite matching at each threshold."
        ),
        "threshold_results": threshold_results,
        "best_candidate_per_candidate": best_per_candidate,
        "best_candidate_per_ground_truth": gt_best_results,
    }


# ============================================================
# STAGE-BY-STAGE POINT DIAGNOSTICS
# ============================================================
def diagnose_ground_truth_points(
    ground_truth: list[dict[str, Any]],
    raw_points: np.ndarray,
    non_ground_points: np.ndarray,
    roi_points: np.ndarray,
) -> list[dict[str, Any]]:
    """Count points inside each GT box before and after each preprocessing stage."""
    diagnostics: list[dict[str, Any]] = []
    for gt in ground_truth:
        box = gt["box"]
        raw_count = count_points_inside(box, raw_points)
        non_ground_count = count_points_inside(box, non_ground_points)
        roi_count = count_points_inside(box, roi_points)
        diagnostics.append(
            {
                "gt_token": gt["token"],
                "label": gt["label"],
                "category": gt["category"],
                "raw_points_in_box": raw_count,
                "after_ground_removal": non_ground_count,
                "after_roi_filter": roi_count,
                "points_removed_by_ground_filter": max(0, raw_count - non_ground_count),
                "points_removed_by_roi_filter": max(0, non_ground_count - roi_count),
            }
        )
    return diagnostics

def diagnose_candidate_coverage(
    candidates: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Diagnose how well generated clusters cover each GT object's LiDAR points."""
    candidate_points_data = [
        (
            candidate_id(candidate, index),
            load_points(candidate_points_path(candidate)),
        )
        for index, candidate in enumerate(candidates)
    ]

    diagnostics: list[dict[str, Any]] = []

    for gt in ground_truth:
        gt_box = gt["box"]
        gt_points_count = count_points_inside(
            gt_box,
            load_points(ROI_POINTS_PATH),
        )

        cluster_intersections: list[int] = []

        for _candidate_id, candidate_points in candidate_points_data:
            intersection = count_points_inside(gt_box, candidate_points)

            if intersection > 0:
                cluster_intersections.append(intersection)

        total_covered = min(
            gt_points_count,
            sum(cluster_intersections),
        )

        diagnostics.append(
            {
                "gt_token": gt["token"],
                "label": gt["label"],
                "category": gt["category"],
                "points_in_roi": gt_points_count,
                "overlapping_clusters": len(cluster_intersections),
                "best_cluster_points": max(cluster_intersections, default=0),
                "covered_points": total_covered,
                "point_coverage": (
                    total_covered / gt_points_count
                    if gt_points_count > 0
                    else 0.0
                ),
            }
        )

    return diagnostics


# ============================================================
# POINT-OVERLAP EVALUATION
# ============================================================
def evaluate_candidates_by_points(
    roi_points: np.ndarray,
    candidates: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    gt_stage_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Point-overlap diagnostic; eligible GT matches are selected after applying thresholds."""
    gt_point_counts = [count_points_inside(gt["box"], roi_points) for gt in ground_truth]
    candidate_results: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        cid = candidate_id(candidate, index)
        candidate_points = load_points(candidate_points_path(candidate))
        point_count = len(candidate_points)
        best_any_match: dict[str, Any] | None = None
        best_eligible_match: dict[str, Any] | None = None

        for gt_index, gt in enumerate(ground_truth):
            gt_count = gt_point_counts[gt_index]
            if point_count == 0 or gt_count == 0:
                continue

            intersection = count_points_inside(gt["box"], candidate_points)
            if intersection == 0:
                continue

            purity = intersection / point_count
            gt_coverage = intersection / gt_count
            union = point_count + gt_count - intersection
            point_iou = intersection / union if union > 0 else 0.0
            match = {
                "gt_token": gt["token"],
                "gt_label": gt["label"],
                "gt_category": gt["category"],
                "intersection_points": int(intersection),
                "candidate_purity": float(purity),
                "gt_point_coverage": float(gt_coverage),
                "point_iou": float(point_iou),
                "eligible_by_thresholds": bool(
                    intersection >= MIN_INTERSECTION_POINTS
                    and purity >= MIN_CANDIDATE_PURITY
                ),
            }

            rank = (purity, intersection, point_iou)
            if best_any_match is None or rank > (
                best_any_match["candidate_purity"],
                best_any_match["intersection_points"],
                best_any_match["point_iou"],
            ):
                best_any_match = match

            # Important: filter by eligibility before selecting the match that
            # determines the candidate's point-overlap label. A non-eligible GT
            # must not hide another GT that satisfies both configured thresholds.
            if match["eligible_by_thresholds"] and (
                best_eligible_match is None
                or rank > (
                    best_eligible_match["candidate_purity"],
                    best_eligible_match["intersection_points"],
                    best_eligible_match["point_iou"],
                )
            ):
                best_eligible_match = match

        matched = best_eligible_match is not None
        report_match = best_eligible_match or best_any_match
        candidate_results.append(
            {
                "candidate_id": cid,
                "points_count": int(point_count),
                "projected_to_camera": is_projected(candidate),
                "matched": matched,
                "predicted_label": (
                    best_eligible_match["gt_label"] if best_eligible_match else None
                ),
                "best_match": report_match,
                "best_eligible_match": best_eligible_match,
            }
        )

    detected_gt_tokens = {
        result["best_eligible_match"]["gt_token"]
        for result in candidate_results
        if result["matched"] and result["best_eligible_match"] is not None
    }

    gt_results: list[dict[str, Any]] = []
    for index, gt in enumerate(ground_truth):
        stage_data = gt_stage_diagnostics[index]
        gt_results.append(
            {
                "gt_token": gt["token"],
                "label": gt["label"],
                "category": gt["category"],
                "center_lidar": [float(v) for v in gt["box"].center],
                "detected": gt["token"] in detected_gt_tokens,
                **stage_data,
                "points_in_roi": gt_point_counts[index],
            }
        )

    by_class: dict[str, dict[str, int]] = {}
    for gt_result in gt_results:
        label = gt_result["label"]
        by_class.setdefault(label, {"total": 0, "detected": 0})
        by_class[label]["total"] += 1
        if gt_result["detected"]:
            by_class[label]["detected"] += 1

    gt_count = len(ground_truth)
    detected_count = len(detected_gt_tokens)
    return {
        "settings": {
            "min_intersection_points": MIN_INTERSECTION_POINTS,
            "min_candidate_purity": MIN_CANDIDATE_PURITY,
            "roi_x": list(ROI_X),
            "roi_y": list(ROI_Y),
            "roi_z": list(ROI_Z),
            "target_classes": [
                "car", "truck", "bus", "motorcycle", "bicycle",
                "special_machinery", "pedestrian", "barrier", "traffic_cone",
            ],
        },
        "summary": {
            "points_after_roi": int(len(roi_points)),
            "ground_truth_objects": gt_count,
            "ground_truth_objects_with_raw_points": sum(
                item["raw_points_in_box"] > 0 for item in gt_results
            ),
            "ground_truth_objects_with_points_after_ground_removal": sum(
                item["after_ground_removal"] > 0 for item in gt_results
            ),
            "ground_truth_objects_with_points_after_roi": sum(
                item["points_in_roi"] > 0 for item in gt_results
            ),
            "detected_ground_truth_objects": detected_count,
            "unmatched_ground_truth_objects": gt_count - detected_count,
            "candidate_clusters": len(candidates),
            "projected_candidates": sum(is_projected(c) for c in candidates),
            "matched_candidates": sum(result["matched"] for result in candidate_results),
            "unmatched_candidates": sum(not result["matched"] for result in candidate_results),
            "ground_truth_recall": detected_count / gt_count if gt_count else 0.0,
            "ground_truth_by_class": by_class,
        },
        "ground_truth": gt_results,
        "candidates": candidate_results,
    }


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    print("Loading candidate metadata...")
    data, candidates = load_candidates()
    sample_token = get_sample_token(data)
    print(f"Sample token: {sample_token}")
    print(f"Candidate clusters: {len(candidates)}")

    raw_points = load_points(RAW_POINTS_PATH)
    non_ground_points = load_points(NON_GROUND_POINTS_PATH)
    roi_points = load_points(ROI_POINTS_PATH)
    print(f"Raw LiDAR points: {len(raw_points)}")
    print(f"After ground removal: {len(non_ground_points)}")
    print(f"After ROI filter: {len(roi_points)}")

    print("Loading nuScenes ground truth...")
    nusc = NuScenes(version=VERSION, dataroot=str(DATA_ROOT), verbose=False)
    ground_truth = load_ground_truth(nusc, sample_token)
    print(f"Visible target GT objects in ROI: {len(ground_truth)}")
    print("Target classes include vehicles, pedestrians, barriers and traffic cones.")

    print("Counting GT points through preprocessing stages...")
    stage_diagnostics = diagnose_ground_truth_points(
        ground_truth, raw_points, non_ground_points, roi_points
    )
    
    print("\nDiagnosing candidate coverage...")
    coverage_diagnostics = diagnose_candidate_coverage(
        candidates,
        ground_truth,
    )

    print("\nGT CANDIDATE COVERAGE")
    print("-" * 100)
    print(
        f"{'CLASS':16s} {'ROI PTS':>8s} {'CLUSTERS':>9s} "
        f"{'BEST PTS':>9s} {'COVERAGE':>10s}"
    )

    for item in coverage_diagnostics:
        print(
            f"{item['label']:16s} "
            f"{item['points_in_roi']:8d} "
            f"{item['overlapping_clusters']:9d} "
            f"{item['best_cluster_points']:9d} "
            f"{item['point_coverage']:10.1%}"
        )


    print("\nGT POINT COUNTS BY STAGE")
    print("-" * 104)
    print(
        f"{'CLASS':16s} {'RAW':>7s} {'NO_GROUND':>11s} "
        f"{'ROI':>7s} {'GROUND_REMOVED':>15s} {'ROI_REMOVED':>12s}"
    )
    for item in stage_diagnostics:
        print(
            f"{item['label']:16s} "
            f"{item['raw_points_in_box']:7d} "
            f"{item['after_ground_removal']:11d} "
            f"{item['after_roi_filter']:7d} "
            f"{item['points_removed_by_ground_filter']:15d} "
            f"{item['points_removed_by_roi_filter']:12d}"
        )

    print("\nEvaluating candidates by point overlap...")
    point_report = evaluate_candidates_by_points(
        roi_points, candidates, ground_truth, stage_diagnostics
    )
    point_report["summary"]["raw_lidar_points"] = int(len(raw_points))
    point_report["summary"]["points_after_ground_removal"] = int(len(non_ground_points))

    print("Evaluating candidate 3D boxes by oriented 3D IoU...")
    box_report = evaluate_boxes_by_iou(candidates, ground_truth)
    report: dict[str, Any] = {
        "sample_token": sample_token,
        "point_overlap_evaluation": point_report,
        "box_iou_evaluation": box_report,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    point_summary = point_report["summary"]
    print("\n" + "=" * 64)
    print("POINT-OVERLAP EVALUATION (DIAGNOSTIC)")
    print("=" * 64)
    print(f"Raw LiDAR points: {point_summary['raw_lidar_points']}")
    print(f"After ground removal: {point_summary['points_after_ground_removal']}")
    print(f"After ROI filter: {point_summary['points_after_roi']}")
    print(f"Visible target GT objects: {point_summary['ground_truth_objects']}")
    print(f"Detected by point overlap: {point_summary['detected_ground_truth_objects']}")
    print(f"Point-overlap recall: {point_summary['ground_truth_recall']:.4f}")
    print("By class:")
    for label, values in sorted(point_summary["ground_truth_by_class"].items()):
        print(f"  {label}: {values['detected']} / {values['total']}")

    print("\n" + "=" * 64)
    print("3D BOX EVALUATION (MAXIMUM-CARDINALITY ONE-TO-ONE MATCHING)")
    print("=" * 64)
    for threshold, result in box_report["threshold_results"].items():
        print(
            f"3D IoU >= {float(threshold):.2f}: "
            f"matched {result['matched_gt_objects']} / {result['total_gt_objects']}, "
            f"recall={result['recall']:.4f}, "
            f"matched candidates={result['matched_candidates']}"
        )
        print("  By class:")
        for label, values in sorted(result["by_class"].items()):
            print(f"    {label}: {values['detected']} / {values['total']}")

    print("\nBest 3D IoU and geometry per candidate:")
    for item in box_report["best_candidate_per_candidate"]:
        best = item["best_match"]
        if best is None:
            print(f"  {item['candidate_id']}: no 3D overlap")
            continue
        print(
            f"  {item['candidate_id']}: "
            f"IoU={best['iou_3d']:.3f}, "
            f"class={best['gt_label']}, "
            f"center_distance={best['center_distance_m']:.2f} m, "
            f"candidate LWH={np.round(best['candidate_dimensions_lwh'], 2).tolist()}, "
            f"GT LWH={np.round(best['gt_dimensions_lwh'], 2).tolist()}, "
            f"major_axis_diff={best['major_axis_difference_deg']:.1f} deg"
        )

    print(f"\nFull report saved to: {OUTPUT_PATH.resolve()}")
    if int(data.get("parameters", {}).get("nsweeps", 1)) > 1:
        print(
            "NOTE: point counts use accumulated LiDAR sweeps. Moving-object counts "
            "are approximate because older sweeps may not align perfectly with current-frame annotations."
        )


if __name__ == "__main__":
    main()
