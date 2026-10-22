"""Run pretrained MMDetection3D PointPillars (nuScenes) on one LiDAR frame.

Runs inside the WSL env (~/pp/.venv). Writes predictions in the LiDAR_TOP
frame using the same box fields as ground_truth_boxes.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mmdet3d
import numpy as np
import torch
from mmdet3d.apis import inference_detector, init_model


# ============================================================
# CONFIGURATION
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    Path(mmdet3d.__file__).parent
    / ".mim"
    / "configs"
    / "pointpillars"
    / "pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
)
CHECKPOINT_PATH = (
    Path.home()
    / "pp"
    / "checkpoints"
    / "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
)
POINTS_PATH = REPO_ROOT / "outputs" / "pointpillars_input" / "frame_010.bin"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pointpillars_nus"
SAMPLE_TOKEN = "023c4df2d451409881d8e6ea82f14704"

# nuScenes raw .pcd.bin layout: x, y, z, intensity, ring_index
LOAD_DIM = 5


# ============================================================
# HELPERS
# ============================================================
def yaw_to_wxyz(yaw: float) -> list[float]:
    """Rotation about +z as a (w, x, y, z) quaternion, nuScenes order."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def describe_pipeline(cfg: Any) -> list[dict[str, Any]]:
    pipeline = cfg.test_dataloader.dataset.pipeline
    return [dict(step) for step in pipeline]


def predictions_to_records(result: Any, class_names: list[str]) -> list[dict[str, Any]]:
    instances = result.pred_instances_3d
    boxes = instances.bboxes_3d
    centers = boxes.gravity_center.cpu().numpy()
    dims = boxes.dims.cpu().numpy()  # (dx, dy, dz) == (l, w, h)
    yaws = boxes.yaw.cpu().numpy()
    raw = boxes.tensor.cpu().numpy()
    scores = instances.scores_3d.cpu().numpy()
    labels = instances.labels_3d.cpu().numpy()

    records = []
    for i in np.argsort(-scores):
        length, width, height = (float(v) for v in dims[i])
        yaw = float(yaws[i])
        record = {
            "label": class_names[int(labels[i])],
            "score": float(scores[i]),
            "center_xyz_m": [float(v) for v in centers[i]],
            "size_wlh_m": [width, length, height],
            "yaw_rad": yaw,
            "orientation_wxyz": yaw_to_wxyz(yaw),
            "mmdet3d_box": [float(v) for v in raw[i]],
        }
        if raw.shape[1] >= 9:
            record["velocity_xy_mps"] = [float(raw[i, 7]), float(raw[i, 8])]
        records.append(record)
    return records


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, default=POINTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    points = np.fromfile(args.points, dtype=np.float32)
    if points.size % LOAD_DIM != 0:
        raise ValueError(f"{args.points}: {points.size} floats is not divisible by {LOAD_DIM}")
    points = points.reshape(-1, LOAD_DIM)
    print(f"Points: {len(points)} x {LOAD_DIM} from {args.points}")

    model = init_model(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=args.device)
    class_names = list(model.dataset_meta["classes"])
    pipeline = describe_pipeline(model.cfg)
    print("Test pipeline:")
    for step in pipeline:
        print("  ", step)

    result, _ = inference_detector(model, str(args.points))
    records = predictions_to_records(result, class_names)
    print(f"Predictions: {len(records)} (model score_thr="
          f"{model.cfg.model.test_cfg.pts.score_thr})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.json"
    payload = {
        "sample_token": SAMPLE_TOKEN,
        "frame": "LIDAR_TOP",
        "points_path": str(args.points),
        "num_input_points": int(len(points)),
        "num_sweeps_used": 1,
        "config": CONFIG_PATH.name,
        "checkpoint": CHECKPOINT_PATH.name,
        "class_names": class_names,
        "box_convention": "center = gravity center; size_wlh = (w, l, h); yaw about +z, 0 = +x",
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mmdet3d": mmdet3d.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "test_pipeline": [str(step) for step in pipeline],
        "predictions": records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved: {output_path}")

    for record in records[:15]:
        x, y, z = record["center_xyz_m"]
        w, l, h = record["size_wlh_m"]
        print(f"  {record['label']:<22} score={record['score']:.3f} "
              f"center=({x:6.2f},{y:6.2f},{z:5.2f}) wlh=({w:.2f},{l:.2f},{h:.2f}) "
              f"yaw={record['yaw_rad']:+.2f}")


if __name__ == "__main__":
    main()
