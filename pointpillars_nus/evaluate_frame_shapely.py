"""Standalone one-frame PointPillars evaluation using Shapely (no MMDetection3D).

Use the devkit-exported GT JSON with num_lidar_pts/num_radar_pts and prediction
JSON files whose predictions and GT are both expressed in the same LIDAR_TOP frame.
This is a diagnostic, not the official nuScenes mAP/NDS evaluator.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "outputs" / "pointpillars_input"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pointpillars_nus"
POINTS_PATH = INPUT_DIR / "frame_010.bin"
GT_PATH = INPUT_DIR / "ground_truth_boxes_devkit.json"
CLASS_RANGE = {
    "car": 50.0, "truck": 50.0, "bus": 50.0, "trailer": 50.0,
    "construction_vehicle": 50.0, "pedestrian": 40.0, "motorcycle": 40.0,
    "bicycle": 40.0, "traffic_cone": 30.0, "barrier": 30.0,
}
SCORE_THRESHOLDS = (0.1, 0.3, 0.5)
CENTER_DIST_THRESHOLD_M = 2.0
IOU_THRESHOLDS = (0.25, 0.5)
MAIN_SCORE_THRESHOLD = 0.3


def category_to_detection_class(category: str) -> str | None:
    mapping = {
        "vehicle.car": "car", "vehicle.truck": "truck",
        "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus",
        "vehicle.trailer": "trailer", "vehicle.construction": "construction_vehicle",
        "vehicle.bicycle": "bicycle", "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone", "movable_object.barrier": "barrier",
    }
    return mapping.get(category)


def wxyz_to_yaw(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def center_xy(r: dict[str, Any]) -> tuple[float, float]:
    return float(r["center_xyz_m"][0]), float(r["center_xyz_m"][1])


def bev_polygon(r: dict[str, Any]) -> Polygon:
    cx, cy, _ = r["center_xyz_m"]
    w, l, _ = r["size_wlh_m"]
    yaw = float(r["yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    local = ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))
    pts = [(cx + c * x - s * y, cy + s * x + c * y) for x, y in local]
    poly = Polygon(pts)
    return poly if poly.is_valid else poly.buffer(0)


def iou3d(a: dict[str, Any], b: dict[str, Any], poly_a: Polygon | None = None,
          poly_b: Polygon | None = None) -> float:
    pa = poly_a if poly_a is not None else bev_polygon(a)
    pb = poly_b if poly_b is not None else bev_polygon(b)
    area_inter = pa.intersection(pb).area
    if area_inter <= 0:
        return 0.0
    za, zb = float(a["center_xyz_m"][2]), float(b["center_xyz_m"][2])
    ha, hb = float(a["size_wlh_m"][2]), float(b["size_wlh_m"][2])
    overlap_h = max(0.0, min(za + ha / 2, zb + hb / 2) - max(za - ha / 2, zb - hb / 2))
    inter = area_inter * overlap_h
    if inter <= 0:
        return 0.0
    wa, la, ha = map(float, a["size_wlh_m"])
    wb, lb, hb = map(float, b["size_wlh_m"])
    union = wa * la * ha + wb * lb * hb - inter
    return float(inter / union) if union > 0 else 0.0


def points_in_boxes(points: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(len(records), dtype=int)
    xyz = points[:, :3]
    for i, r in enumerate(records):
        cx, cy, cz = map(float, r["center_xyz_m"])
        w, l, h = map(float, r["size_wlh_m"])
        c, s = math.cos(float(r["yaw"])), math.sin(float(r["yaw"]))
        dx, dy, dz = xyz[:, 0] - cx, xyz[:, 1] - cy, xyz[:, 2] - cz
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        counts[i] = int(((np.abs(local_x) <= l / 2) & (np.abs(local_y) <= w / 2) & (np.abs(dz) <= h / 2)).sum())
    return counts


def bev_distance(center: list[float], lidar2ego: np.ndarray) -> float:
    ego = lidar2ego[:3, :3] @ np.asarray(center, dtype=float) + lidar2ego[:3, 3]
    return math.hypot(float(ego[0]), float(ego[1]))


def summarize(num_pred: int, num_gt: int, num_tp: int) -> dict[str, Any]:
    return {
        "predictions": num_pred, "ground_truth": num_gt, "tp": num_tp,
        "fp": num_pred - num_tp, "fn": num_gt - num_tp,
        "precision": num_tp / num_pred if num_pred else None,
        "recall": num_tp / num_gt if num_gt else None,
    }


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def greedy_match(preds: list[dict[str, Any]], gts: list[dict[str, Any]],
                 iou: np.ndarray, criterion: str, threshold: float) -> list[tuple[int, int]]:
    matched_gt: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for pi in sorted(range(len(preds)), key=lambda i: -float(preds[i]["score"])):
        best_gi, best_value = -1, None
        for gi, gt in enumerate(gts):
            if gi in matched_gt or gt["label"] != preds[pi]["label"]:
                continue
            if criterion == "center":
                value = math.dist(center_xy(preds[pi]), center_xy(gt))
                ok = value <= threshold and (best_value is None or value < best_value)
            else:
                value = float(iou[pi, gi])
                ok = value >= threshold and (best_value is None or value > best_value)
            if ok:
                best_gi, best_value = gi, value
        if best_gi >= 0:
            matched_gt.add(best_gi)
            pairs.append((pi, best_gi))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=GT_PATH)
    parser.add_argument("--points", type=Path, default=POINTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", required=True, help="e.g. 1sweep_shapely or 10sweeps_shapely")
    args = parser.parse_args()

    with args.gt.open("r", encoding="utf-8-sig") as f:
        gt_raw = json.load(f)
    with args.predictions.open("r", encoding="utf-8-sig") as f:
        pred_payload = json.load(f)

    lidar2ego_raw = pred_payload.get("lidar2ego")
    if lidar2ego_raw is None:
        raise ValueError("Predictions file has no lidar2ego. Use aligned predictions for the 1-sweep run.")
    lidar2ego = np.asarray(lidar2ego_raw, dtype=float)
    if lidar2ego.shape != (4, 4):
        raise ValueError(f"Expected lidar2ego shape (4,4), got {lidar2ego.shape}")

    gts_all: list[dict[str, Any]] = []
    unmapped: dict[str, int] = {}
    for g in gt_raw:
        label = category_to_detection_class(g["category"])
        if label is None:
            unmapped[g["category"]] = unmapped.get(g["category"], 0) + 1
            continue
        if "num_lidar_pts" not in g or "num_radar_pts" not in g:
            raise ValueError("GT must be the devkit-exported JSON containing num_lidar_pts and num_radar_pts.")
        gts_all.append({**g, "label": label, "yaw": wxyz_to_yaw(g["orientation_wxyz"])})

    # This count is diagnostic only; official GT inclusion uses devkit point counts below.
    points = np.fromfile(args.points, dtype=np.float32).reshape(-1, 5)
    keyframe_counts = points_in_boxes(points, gts_all)
    for g in gts_all:
        g["distance_m"] = bev_distance(g["center_xyz_m"], lidar2ego)
        g["has_points"] = int(g["num_lidar_pts"]) + int(g["num_radar_pts"]) > 0
    gts = [g for g in gts_all if g["distance_m"] <= CLASS_RANGE[g["label"]] and g["has_points"]]

    preds_all: list[dict[str, Any]] = []
    for p in pred_payload["predictions"]:
        if "yaw_rad" not in p:
            raise ValueError("Prediction record is missing yaw_rad.")
        pp = {**p, "yaw": float(p["yaw_rad"])}
        pp["distance_m"] = bev_distance(pp["center_xyz_m"], lidar2ego)
        preds_all.append(pp)
    preds_in_range = [p for p in preds_all if p["distance_m"] <= CLASS_RANGE[p["label"]]]

    # Compute each oriented 3D IoU once and reuse for all score/IoU thresholds.
    pred_polys = [bev_polygon(p) for p in preds_in_range]
    gt_polys = [bev_polygon(g) for g in gts]
    iou = np.zeros((len(preds_in_range), len(gts)), dtype=np.float32)
    for pi, pred in enumerate(preds_in_range):
        for gi, gt in enumerate(gts):
            iou[pi, gi] = iou3d(pred, gt, pred_polys[pi], gt_polys[gi])

    report: dict[str, Any] = {
        "note": "Single-frame diagnostic using Shapely polygon intersection; not official nuScenes mAP/NDS.",
        "sample_token": pred_payload.get("sample_token"),
        "num_sweeps_used": pred_payload.get("num_sweeps_used"),
        "gt_total_records": len(gt_raw),
        "gt_unmapped_categories": unmapped,
        "gt_mapped": len(gts_all),
        "gt_mapped_with_keyframe_points": int((keyframe_counts > 0).sum()),
        "gt_evaluated": len(gts),
        "gt_evaluated_per_class": {},
        "eval_region": "per-class BEV range from ego origin; GT with num_lidar_pts + num_radar_pts > 0",
        "predictions_total": len(preds_all),
        "predictions_in_range": len(preds_in_range),
        "results": [],
    }
    for g in gts:
        report["gt_evaluated_per_class"][g["label"]] = report["gt_evaluated_per_class"].get(g["label"], 0) + 1

    for score_thr in SCORE_THRESHOLDS:
        # Preserve prediction indices so the precomputed IoU matrix remains valid.
        keep_indices = [i for i, p in enumerate(preds_in_range) if float(p["score"]) >= score_thr]
        preds = [preds_in_range[i] for i in keep_indices]
        iou_thr = iou[keep_indices, :] if keep_indices else np.zeros((0, len(gts)), dtype=np.float32)
        criteria = [("center", CENTER_DIST_THRESHOLD_M)] + [("iou3d", t) for t in IOU_THRESHOLDS]
        for criterion, threshold in criteria:
            pairs = greedy_match(preds, gts, iou_thr, criterion, threshold)
            entry: dict[str, Any] = {
                "score_thr": score_thr, "criterion": criterion, "threshold": threshold,
                **summarize(len(preds), len(gts), len(pairs)),
            }
            if pairs:
                entry["mean_center_error_m"] = float(np.mean([
                    math.dist(center_xy(preds[p]), center_xy(gts[g])) for p, g in pairs]))
                entry["mean_iou3d"] = float(np.mean([iou_thr[p, g] for p, g in pairs]))
                entry["mean_yaw_error_rad"] = float(np.mean([
                    angle_diff(preds[p]["yaw"], gts[g]["yaw"]) for p, g in pairs]))
            per_class: dict[str, Any] = {}
            labels = sorted({g["label"] for g in gts} | {p["label"] for p in preds})
            for label in labels:
                n_pred = sum(p["label"] == label for p in preds)
                n_gt = sum(g["label"] == label for g in gts)
                n_tp = sum(gts[g]["label"] == label for _, g in pairs)
                per_class[label] = summarize(n_pred, n_gt, n_tp)
            entry["per_class"] = per_class
            report["results"].append(entry)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"evaluation_frame_{args.tag}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Sample: {report['sample_token']} | sweeps={report['num_sweeps_used']}")
    print(f"GT: records={len(gt_raw)}, mapped={len(gts_all)}, evaluated={len(gts)} {report['gt_evaluated_per_class']}")
    print(f"Predictions: total={len(preds_all)}, in range={len(preds_in_range)}")
    print(f"{'score>=':>7} {'criterion':<14} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'IoU':>6} {'dXY':>6}")
    for e in report["results"]:
        crit = f"{e['criterion']}{'<=' if e['criterion'] == 'center' else '>='}{e['threshold']}"
        p = f"{e['precision']:.3f}" if e["precision"] is not None else "-"
        r = f"{e['recall']:.3f}" if e["recall"] is not None else "-"
        print(f"{e['score_thr']:>7.1f} {crit:<14} {e['predictions']:>5} {e['tp']:>4} {e['fp']:>4} {e['fn']:>4} {p:>6} {r:>6} {e.get('mean_iou3d', 0):>6.3f} {e.get('mean_center_error_m', 0):>6.3f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
