"""Single-frame diagnostic evaluation of PointPillars predictions against GT.

Both inputs are boxes in the LIDAR_TOP frame (center, wlh, wxyz quaternion).
This is a one-frame diagnostic, not the official nuScenes mAP/NDS.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmdet3d.structures import LiDARInstance3DBoxes


# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "outputs" / "pointpillars_input"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pointpillars_nus"
POINTS_PATH = INPUT_DIR / "frame_010.bin"
GT_PATH = INPUT_DIR / "ground_truth_boxes.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.json"

# Official nuScenes detection eval class ranges (metres, BEV distance).
CLASS_RANGE = {
    "car": 50.0,
    "truck": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "construction_vehicle": 50.0,
    "pedestrian": 40.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
    "traffic_cone": 30.0,
    "barrier": 30.0,
}
SCORE_THRESHOLDS = (0.1, 0.3, 0.5)
CENTER_DIST_THRESHOLD_M = 2.0  # nuScenes TP criterion (one of 0.5/1/2/4)
IOU_THRESHOLDS = (0.25, 0.5)
MAIN_SCORE_THRESHOLD = 0.3


# ============================================================
# CATEGORY MAPPING
# ============================================================
def category_to_detection_class(category: str) -> str | None:
    """Official nuScenes category -> detection class mapping."""
    mapping = {
        "vehicle.car": "car",
        "vehicle.truck": "truck",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.trailer": "trailer",
        "vehicle.construction": "construction_vehicle",
        "vehicle.bicycle": "bicycle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "movable_object.barrier": "barrier",
    }
    return mapping.get(category)


# ============================================================
# GEOMETRY
# ============================================================
def wxyz_to_yaw(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_mmdet3d_boxes(records: list[dict[str, Any]]) -> LiDARInstance3DBoxes:
    """(center, wlh, yaw) -> mmdet3d LiDAR boxes (bottom center, l, w, h, yaw)."""
    rows = []
    for r in records:
        x, y, z = r["center_xyz_m"]
        w, l, h = r["size_wlh_m"]
        rows.append([x, y, z, l, w, h, r["yaw"]])
    tensor = torch.tensor(rows, dtype=torch.float32).reshape(-1, 7)
    return LiDARInstance3DBoxes(tensor, box_dim=7, origin=(0.5, 0.5, 0.5))


def points_in_boxes(points: np.ndarray, records: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(len(records), dtype=int)
    xyz = points[:, :3]
    for i, r in enumerate(records):
        cx, cy, cz = r["center_xyz_m"]
        w, l, h = r["size_wlh_m"]
        c, s = math.cos(r["yaw"]), math.sin(r["yaw"])
        dx, dy, dz = xyz[:, 0] - cx, xyz[:, 1] - cy, xyz[:, 2] - cz
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        inside = (np.abs(local_x) <= l / 2) & (np.abs(local_y) <= w / 2) & (np.abs(dz) <= h / 2)
        counts[i] = int(inside.sum())
    return counts


def bev_distance(center: list[float], lidar2ego: np.ndarray | None) -> float:
    """BEV distance from the ego origin (official) or from LIDAR_TOP if no calib."""
    if lidar2ego is None:
        return math.hypot(center[0], center[1])
    ego = lidar2ego[:3, :3] @ np.asarray(center) + lidar2ego[:3, 3]
    return math.hypot(ego[0], ego[1])


def box_corners_bev(r: dict[str, Any]) -> np.ndarray:
    cx, cy, _ = r["center_xyz_m"]
    w, l, _ = r["size_wlh_m"]
    c, s = math.cos(r["yaw"]), math.sin(r["yaw"])
    local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def box_corners_3d(r: dict[str, Any]) -> np.ndarray:
    bev = box_corners_bev(r)
    z = r["center_xyz_m"][2]
    h = r["size_wlh_m"][2]
    bottom = np.column_stack([bev, np.full(4, z - h / 2)])
    top = np.column_stack([bev, np.full(4, z + h / 2)])
    return np.vstack([bottom, top])


# ============================================================
# MATCHING
# ============================================================
def greedy_match(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou: np.ndarray,
    criterion: str,
    threshold: float,
) -> list[tuple[int, int]]:
    """One-to-one, same-class, highest-score-first matching (nuScenes style)."""
    matched_gt: set[int] = set()
    pairs = []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        best_gi, best_value = -1, None
        for gi, gt in enumerate(gts):
            if gi in matched_gt or gt["label"] != preds[pi]["label"]:
                continue
            if criterion == "center":
                value = math.dist(preds[pi]["center_xyz_m"][:2], gt["center_xyz_m"][:2])
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


def summarize(num_pred: int, num_gt: int, num_tp: int) -> dict[str, Any]:
    return {
        "predictions": num_pred,
        "ground_truth": num_gt,
        "tp": num_tp,
        "fp": num_pred - num_tp,
        "fn": num_gt - num_tp,
        "precision": num_tp / num_pred if num_pred else None,
        "recall": num_tp / num_gt if num_gt else None,
    }


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


# ============================================================
# VISUALIZATION
# ============================================================
def plot_bev(points, gts, preds, tp_pred, tp_gt, path: Path, score_thr: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 12))
    keep = (np.abs(points[:, 0]) < 55) & (np.abs(points[:, 1]) < 55)
    ax.scatter(points[keep, 0], points[keep, 1], s=0.2, c="0.6", linewidths=0)
    for gi, gt in enumerate(gts):
        poly = np.vstack([box_corners_bev(gt), box_corners_bev(gt)[:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="limegreen" if gi in tp_gt else "darkorange", lw=1.6)
    for pi, pred in enumerate(preds):
        corners = box_corners_bev(pred)
        poly = np.vstack([corners, corners[:1]])
        ax.plot(poly[:, 0], poly[:, 1], color="royalblue" if pi in tp_pred else "red",
                lw=1.0, ls="--")
        front = corners[:2].mean(axis=0)
        cx, cy, _ = pred["center_xyz_m"]
        ax.plot([cx, front[0]], [cy, front[1]], color="royalblue" if pi in tp_pred else "red", lw=0.8)
        ax.text(cx, cy, f"{pred['label'][:4]} {pred['score']:.2f}", fontsize=5)
    for radius in (30, 40, 50):
        ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="0.8", lw=0.6, ls=":"))
    ax.set_aspect("equal")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.set_xlabel("x, m (LIDAR_TOP)")
    ax.set_ylabel("y, m (LIDAR_TOP)")
    ax.set_title(
        f"PointPillars nuScenes (score>={score_thr}, match: center<={CENTER_DIST_THRESHOLD_M} m)\n"
        "GT matched: green | GT missed: orange | Pred TP: blue dashed | Pred FP: red dashed"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d(points, gts, preds, path: Path) -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(projection="3d")
    keep = (np.abs(points[:, 0]) < 40) & (np.abs(points[:, 1]) < 40)
    sub = points[keep][::2]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], s=0.1, c=sub[:, 2], cmap="viridis", linewidths=0)
    for records, color in ((gts, "limegreen"), (preds, "red")):
        for r in records:
            if max(abs(r["center_xyz_m"][0]), abs(r["center_xyz_m"][1])) > 40:
                continue
            corners = box_corners_3d(r)
            for a, b in edges:
                ax.plot(*zip(corners[a], corners[b]), color=color, lw=0.8)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-5, 5)
    ax.set_box_aspect((8, 8, 1))
    ax.view_init(elev=35, azim=-60)
    ax.set_title("3D view (|x|,|y| < 40 m): GT green, predictions red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--gt", type=Path, default=GT_PATH)
    parser.add_argument("--points", type=Path, default=POINTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--tag", default="", help="suffix for output file names")
    args = parser.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    points = np.fromfile(args.points, dtype=np.float32).reshape(-1, 5)
    with args.gt.open("r", encoding="utf-8") as file:
        gt_raw = json.load(file)
    with args.predictions.open("r", encoding="utf-8") as file:
        pred_payload = json.load(file)

    # ---- GT: map classes, yaw, frame sanity check via points inside boxes
    gts_all = []
    unmapped: dict[str, int] = {}
    for g in gt_raw:
        label = category_to_detection_class(g["category"])
        if label is None:
            unmapped[g["category"]] = unmapped.get(g["category"], 0) + 1
            continue
        gts_all.append({**g, "label": label, "yaw": wxyz_to_yaw(g["orientation_wxyz"])})
    lidar2ego = np.asarray(pred_payload["lidar2ego"]) if "lidar2ego" in pred_payload else None
    official_counts = all("num_lidar_pts" in g for g in gts_all)
    gt_point_counts = points_in_boxes(points, gts_all)
    for g, n in zip(gts_all, gt_point_counts):
        g["num_lidar_points"] = int(n)
        g["distance_m"] = bev_distance(g["center_xyz_m"], lidar2ego)
        # Official filter: num_lidar_pts + num_radar_pts > 0; else keyframe points in box.
        g["has_points"] = (
            g["num_lidar_pts"] + g["num_radar_pts"] > 0 if official_counts else n > 0
        )

    gts = [g for g in gts_all if g["distance_m"] <= CLASS_RANGE[g["label"]] and g["has_points"]]

    # ---- Predictions
    preds_all = [{**p, "yaw": p["yaw_rad"]} for p in pred_payload["predictions"]]
    for p in preds_all:
        p["distance_m"] = bev_distance(p["center_xyz_m"], lidar2ego)
    preds_in_range = [p for p in preds_all if p["distance_m"] <= CLASS_RANGE[p["label"]]]

    report: dict[str, Any] = {
        "note": "Single-frame diagnostic; not the official nuScenes mAP/NDS.",
        "sample_token": pred_payload.get("sample_token"),
        "num_sweeps_used": pred_payload.get("num_sweeps_used"),
        "gt_total_records": len(gt_raw),
        "gt_unmapped_categories": unmapped,
        "gt_mapped": len(gts_all),
        "gt_mapped_with_points": int((gt_point_counts > 0).sum()),
        "gt_evaluated": len(gts),
        "gt_evaluated_per_class": {},
        "eval_region": (
            f"official per-class BEV range from {'ego' if lidar2ego is not None else 'LIDAR_TOP'} origin; "
            + ("GT with num_lidar_pts + num_radar_pts > 0 (official)" if official_counts
               else "GT with >=1 keyframe LiDAR point")
        ),
        "predictions_total": len(preds_all),
        "predictions_in_range": len(preds_in_range),
        "results": [],
    }
    for g in gts:
        report["gt_evaluated_per_class"][g["label"]] = report["gt_evaluated_per_class"].get(g["label"], 0) + 1

    main_pairs: list[tuple[int, int]] = []
    main_preds: list[dict[str, Any]] = []
    for score_thr in SCORE_THRESHOLDS:
        preds = [p for p in preds_in_range if p["score"] >= score_thr]
        if preds and gts:
            iou = LiDARInstance3DBoxes.overlaps(to_mmdet3d_boxes(preds), to_mmdet3d_boxes(gts)).numpy()
        else:
            iou = np.zeros((len(preds), len(gts)))
        criteria = [("center", CENTER_DIST_THRESHOLD_M)] + [("iou3d", t) for t in IOU_THRESHOLDS]
        for criterion, threshold in criteria:
            pairs = greedy_match(preds, gts, iou, criterion, threshold)
            entry = {"score_thr": score_thr, "criterion": criterion, "threshold": threshold,
                     **summarize(len(preds), len(gts), len(pairs))}
            if pairs:
                entry["mean_center_error_m"] = float(np.mean(
                    [math.dist(preds[p]["center_xyz_m"][:2], gts[g]["center_xyz_m"][:2]) for p, g in pairs]))
                entry["mean_iou3d"] = float(np.mean([iou[p, g] for p, g in pairs]))
                entry["mean_yaw_error_rad"] = float(np.mean(
                    [angle_diff(preds[p]["yaw"], gts[g]["yaw"]) for p, g in pairs]))
            per_class = {}
            for label in sorted({g["label"] for g in gts} | {p["label"] for p in preds}):
                n_pred = sum(p["label"] == label for p in preds)
                n_gt = sum(g["label"] == label for g in gts)
                n_tp = sum(gts[g]["label"] == label for _, g in pairs)
                per_class[label] = summarize(n_pred, n_gt, n_tp)
            entry["per_class"] = per_class
            report["results"].append(entry)
            if score_thr == MAIN_SCORE_THRESHOLD and criterion == "center":
                main_pairs, main_preds = pairs, preds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"evaluation_frame{suffix}.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    plot_bev(points, gts, main_preds, {p for p, _ in main_pairs}, {g for _, g in main_pairs},
             args.output_dir / f"bev_pred_vs_gt{suffix}.png", MAIN_SCORE_THRESHOLD)
    plot_3d(points, gts, main_preds, args.output_dir / f"view3d_pred_vs_gt{suffix}.png")

    print(f"GT records: {len(gt_raw)} | mapped to 10 classes: {len(gts_all)} | unmapped: {unmapped}")
    print(f"GT with >=1 keyframe LiDAR point inside box: {report['gt_mapped_with_points']}/{len(gts_all)}")
    print(f"Eval region: {report['eval_region']}")
    print(f"GT evaluated: {len(gts)} {report['gt_evaluated_per_class']}")
    print(f"Predictions: total {len(preds_all)}, in class range {len(preds_in_range)}")
    print(f"{'score>=':>7} {'criterion':<14} {'pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'IoU':>5} {'dXY':>5}")
    for e in report["results"]:
        crit = f"{e['criterion']}{'<=' if e['criterion'] == 'center' else '>='}{e['threshold']}"
        p = f"{e['precision']:.3f}" if e["precision"] is not None else "  -  "
        r = f"{e['recall']:.3f}" if e["recall"] is not None else "  -  "
        print(f"{e['score_thr']:>7} {crit:<14} {e['predictions']:>5} {e['tp']:>4} {e['fp']:>4} {e['fn']:>4} "
              f"{p:>6} {r:>6} {e.get('mean_iou3d', 0):>5.2f} {e.get('mean_center_error_m', 0):>5.2f}")
    main = next(e for e in report["results"]
                if e["score_thr"] == MAIN_SCORE_THRESHOLD and e["criterion"] == "center")
    print(f"Per class (score>={MAIN_SCORE_THRESHOLD}, center<={CENTER_DIST_THRESHOLD_M} m):")
    for label, s in main["per_class"].items():
        print(f"  {label:<22} pred={s['predictions']:>3} gt={s['ground_truth']:>3} tp={s['tp']:>3}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
