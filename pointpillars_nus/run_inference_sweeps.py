"""Run pretrained PointPillars (nuScenes) on a keyframe + previous LiDAR sweeps.

Matches the training-time input: up to 10 previous LIDAR_TOP sweeps are moved
into the keyframe LIDAR_TOP frame and column 4 carries the time lag (seconds),
exactly as LoadPointsFromMultiSweeps expects. Also exports devkit GT for the
same sample in the LIDAR_TOP frame with the official num_lidar/radar_pts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from run_inference import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    LOAD_DIM,
    OUTPUT_DIR,
    POINTS_PATH,
    REPO_ROOT,
    SAMPLE_TOKEN,
    predictions_to_records,
)
from mmdet3d.apis import init_model
from mmdet3d.structures import get_box_type


# ============================================================
# CONFIGURATION
# ============================================================
DATA_ROOT = Path.home() / "data" / "nuscenes"
VERSION = "v1.0-mini"
NUM_SWEEPS = 10  # same as the converter's max_sweeps and the config's sweeps_num
GT_DEVKIT_PATH = REPO_ROOT / "outputs" / "pointpillars_input" / "ground_truth_boxes_devkit.json"
GT_LEGACY_PATH = REPO_ROOT / "outputs" / "pointpillars_input" / "ground_truth_boxes.json"


# ============================================================
# TRANSFORMS
# ============================================================
def make_transform(rotation_wxyz: list[float], translation: list[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation_wxyz).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def lidar_to_global(nusc: NuScenes, sample_data: dict[str, Any]) -> np.ndarray:
    cs = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    lidar2ego = make_transform(cs["rotation"], cs["translation"])
    ego2global = make_transform(pose["rotation"], pose["translation"])
    return ego2global @ lidar2ego


def collect_sweeps(nusc: NuScenes, key_sd: dict[str, Any], num_sweeps: int) -> list[dict[str, Any]]:
    """Previous LIDAR_TOP sweeps in mmdet3d v2 info format (lidar2sensor)."""
    key_global_to_lidar = np.linalg.inv(lidar_to_global(nusc, key_sd))
    sweeps = []
    sd = key_sd
    while sd["prev"] and len(sweeps) < num_sweeps:
        sd = nusc.get("sample_data", sd["prev"])
        sweep2key = key_global_to_lidar @ lidar_to_global(nusc, sd)
        rotation, translation = sweep2key[:3, :3], sweep2key[:3, 3]
        # LoadPointsFromMultiSweeps does: p @ lidar2sensor[:3,:3] - lidar2sensor[:3,3]
        # which must equal p @ R.T + t for p_key = R p_sweep + t.
        lidar2sensor = np.eye(4)
        lidar2sensor[:3, :3] = rotation.T
        lidar2sensor[:3, 3] = -translation
        sweeps.append({
            "lidar_points": {
                "lidar_path": str(DATA_ROOT / sd["filename"]),
                "lidar2sensor": lidar2sensor.tolist(),
            },
            "timestamp": sd["timestamp"] / 1e6,
            "sample_data_token": sd["token"],
        })
    return sweeps


# ============================================================
# GROUND TRUTH
# ============================================================
def export_devkit_gt(nusc: NuScenes, key_sd: dict[str, Any]) -> list[dict[str, Any]]:
    _, boxes, _ = nusc.get_sample_data(key_sd["token"])  # LIDAR_TOP frame
    records = []
    for box in boxes:
        ann = nusc.get("sample_annotation", box.token)
        records.append({
            "category": box.name,
            "center_xyz_m": [float(v) for v in box.center],
            "size_wlh_m": [float(v) for v in box.wlh],
            "orientation_wxyz": [float(v) for v in box.orientation.elements],
            "annotation_token": box.token,
            "num_lidar_pts": int(ann["num_lidar_pts"]),
            "num_radar_pts": int(ann["num_radar_pts"]),
            "visibility_token": ann["visibility_token"],
        })
    return records


def compare_with_legacy_gt(devkit: list[dict[str, Any]]) -> None:
    if not GT_LEGACY_PATH.is_file():
        print(f"Legacy GT not found: {GT_LEGACY_PATH}")
        return
    with GT_LEGACY_PATH.open("r", encoding="utf-8") as file:
        legacy = {r["annotation_token"]: r for r in json.load(file)}
    by_token = {r["annotation_token"]: r for r in devkit}
    common = set(legacy) & set(by_token)
    max_center = max(
        (np.abs(np.subtract(legacy[t]["center_xyz_m"], by_token[t]["center_xyz_m"])).max() for t in common),
        default=float("nan"),
    )
    print(f"Legacy GT vs devkit GT (LIDAR_TOP): legacy={len(legacy)} devkit={len(by_token)} "
          f"common={len(common)} max |center diff|={max_center:.2e} m")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, default=DATA_ROOT)
    parser.add_argument("--sample-token", default=SAMPLE_TOKEN)
    parser.add_argument("--num-sweeps", type=int, default=NUM_SWEEPS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    nusc = NuScenes(version=VERSION, dataroot=str(args.dataroot), verbose=False)
    sample = nusc.get("sample", args.sample_token)
    key_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    key_path = args.dataroot / key_sd["filename"]
    scene = nusc.get("scene", sample["scene_token"])
    print(f"Sample {args.sample_token} | scene {scene['name']} | {key_sd['filename']}")

    if POINTS_PATH.is_file():
        same = POINTS_PATH.read_bytes() == key_path.read_bytes()
        print(f"frame_010.bin identical to keyframe file: {same}")

    devkit_gt = export_devkit_gt(nusc, key_sd)
    with GT_DEVKIT_PATH.open("w", encoding="utf-8") as file:
        json.dump(devkit_gt, file, indent=2)
    print(f"Devkit GT: {len(devkit_gt)} boxes -> {GT_DEVKIT_PATH}")
    compare_with_legacy_gt(devkit_gt)

    sweeps = collect_sweeps(nusc, key_sd, args.num_sweeps)
    key_ts = key_sd["timestamp"] / 1e6
    lags = [key_ts - s["timestamp"] for s in sweeps]
    print(f"Sweeps: {len(sweeps)} | time lags {min(lags, default=0):.3f}..{max(lags, default=0):.3f} s")

    init_default_scope("mmdet3d")
    model = init_model(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    class_names = list(model.dataset_meta["classes"])
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    box_type_3d, box_mode_3d = get_box_type(model.cfg.test_dataloader.dataset.box_type_3d)
    data = pipeline({
        "lidar_points": {"lidar_path": str(key_path), "num_pts_feats": LOAD_DIM},
        "timestamp": key_ts,
        "lidar_sweeps": sweeps,
        "axis_align_matrix": np.eye(4),
        "box_type_3d": box_type_3d,
        "box_mode_3d": box_mode_3d,
    })
    model_points = data["inputs"]["points"]
    print(f"Points fed to model: {tuple(model_points.shape)} "
          f"(time column range {model_points[:, 3].min():.3f}..{model_points[:, 3].max():.3f})")
    with torch.no_grad():
        result = model.test_step(pseudo_collate([data]))[0]

    records = predictions_to_records(result, class_names)
    print(f"Predictions: {len(records)}")

    cs = nusc.get("calibrated_sensor", key_sd["calibrated_sensor_token"])
    payload = {
        "sample_token": args.sample_token,
        "scene": scene["name"],
        "frame": "LIDAR_TOP",
        "points_path": str(key_path),
        "num_input_points": int(model_points.shape[0]),
        "num_sweeps_used": len(sweeps),
        "sweep_tokens": [s["sample_data_token"] for s in sweeps],
        "lidar2ego": make_transform(cs["rotation"], cs["translation"]).tolist(),
        "config": CONFIG_PATH.name,
        "checkpoint": CHECKPOINT_PATH.name,
        "class_names": class_names,
        "box_convention": "center = gravity center; size_wlh = (w, l, h); yaw about +z, 0 = +x",
        "predictions": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"predictions_{len(sweeps)}sweeps.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved: {output_path}")
    for record in records[:10]:
        x, y, z = record["center_xyz_m"]
        print(f"  {record['label']:<22} score={record['score']:.3f} center=({x:6.2f},{y:6.2f},{z:5.2f})")


if __name__ == "__main__":
    main()
