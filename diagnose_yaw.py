"""
Диагностика yaw GT-боксов и кандидатов.
Запускается без перезапуска prepare_candidate_sample.py.

Цель: установить реальный GT yaw для барьеров и автомобилей
и сравнить с yaw кандидатов, чтобы определить наличие
систематической ошибки ориентации в footprint().

Запуск:
    python diagnose_yaw.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility
from pyquaternion import Quaternion

DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")
CANDIDATES_PATH = Path("outputs/candidate_sample/scene-0061_candidates.json")
SAMPLE_TOKEN = "ca9a282c9e77460f8360f564131a8af5"


def main() -> None:
    nusc = NuScenes(version="v1.0-mini", dataroot=str(DATA_ROOT), verbose=False)

    sample = nusc.get("sample", SAMPLE_TOKEN)
    lidar_token = sample["data"]["LIDAR_TOP"]
    _, gt_boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=BoxVisibility.NONE)

    with CANDIDATES_PATH.open(encoding="utf-8") as f:
        candidates_data = json.load(f)
    candidates = candidates_data["candidates"]

    # ----------------------------------------------------------------
    # 1. GT yaw для барьеров и автомобилей
    # ----------------------------------------------------------------
    print("=" * 80)
    print("GT BOXES: yaw и размеры в системе LIDAR_TOP")
    print("=" * 80)
    print(f"{'label':25s} {'yaw_deg':>8s} {'wlh[0]W':>8s} {'wlh[1]L':>8s} {'wlh[2]H':>8s}  center_y")
    print("-" * 80)

    interesting = ("barrier", "car", "truck", "pedestrian")
    for box in gt_boxes:
        if not any(k in box.name for k in interesting):
            continue
        yaw_deg = np.degrees(float(box.orientation.yaw_pitch_roll[0]))
        print(
            f"{box.name[:25]:25s} {yaw_deg:8.1f} "
            f"{box.wlh[0]:8.3f} {box.wlh[1]:8.3f} {box.wlh[2]:8.3f}"
            f"  y={box.center[1]:.1f}"
        )

    # ----------------------------------------------------------------
    # 2. Кандидаты: yaw и размеры
    # ----------------------------------------------------------------
    print()
    print("=" * 80)
    print("КАНДИДАТЫ: yaw и размеры (из candidates.json)")
    print("=" * 80)
    print(f"{'id':6s} {'yaw_deg':>8s} {'L':>7s} {'W':>7s} {'H':>7s}  center_y  pts")
    print("-" * 80)

    for c in candidates:
        if c["point_count"] < 3:
            continue
        cid = c["candidate_id"]
        yaw_deg = np.degrees(c["yaw_lidar_rad"])
        lwh = c["dimensions_lidar_lwh"]
        cy = c["center_lidar_xyz"][1]
        pts = c["point_count"]
        print(f"C{cid:02d}    {yaw_deg:8.1f} {lwh[0]:7.3f} {lwh[1]:7.3f} {lwh[2]:7.3f}  y={cy:.1f}  {pts}")

    # ----------------------------------------------------------------
    # 3. Парное сравнение: кандидат vs ближайший GT-барьер
    # ----------------------------------------------------------------
    print()
    print("=" * 80)
    print("ПАРНОЕ СРАВНЕНИЕ кандидат vs GT-барьер (центр-центр < 3 м)")
    print("=" * 80)
    print(f"{'Cand':6s} {'GT_token':10s} {'dist':>6s} "
          f"{'cand_yaw':>9s} {'gt_yaw':>8s} {'diff':>8s} "
          f"{'cand_L':>7s} {'gt_W(wlh0)':>10s} {'gt_L(wlh1)':>10s}")
    print("-" * 90)

    barriers_gt = [b for b in gt_boxes if "barrier" in b.name]
    for c in candidates:
        c_center = np.array(c["center_lidar_xyz"])
        c_yaw = float(c["yaw_lidar_rad"])
        best_dist = float("inf")
        best_gt: Box | None = None
        for gt_box in barriers_gt:
            d = float(np.linalg.norm(c_center - gt_box.center))
            if d < best_dist:
                best_dist = d
                best_gt = gt_box
        if best_dist > 3.0 or best_gt is None:
            continue
        gt_yaw = float(best_gt.orientation.yaw_pitch_roll[0])
        diff = float(((c_yaw - gt_yaw + np.pi / 2) % np.pi) - np.pi / 2)
        cid = c["candidate_id"]
        print(
            f"C{cid:02d}    {best_gt.token[:8]:10s} {best_dist:6.2f} "
            f"{np.degrees(c_yaw):9.1f} {np.degrees(gt_yaw):8.1f} {np.degrees(diff):8.1f} "
            f"{c['dimensions_lidar_lwh'][0]:7.3f} {best_gt.wlh[0]:10.3f} {best_gt.wlh[1]:10.3f}"
        )

    # ----------------------------------------------------------------
    # 4. Парное сравнение для автомобилей
    # ----------------------------------------------------------------
    print()
    print("=" * 80)
    print("ПАРНОЕ СРАВНЕНИЕ кандидат vs GT-автомобиль/грузовик (центр < 5 м)")
    print("=" * 80)
    print(f"{'Cand':6s} {'GT_token':10s} {'dist':>6s} "
          f"{'cand_yaw':>9s} {'gt_yaw':>8s} {'diff':>8s} "
          f"{'cand_L':>7s} {'gt_W':>7s} {'gt_L':>7s}")
    print("-" * 80)

    vehicle_gt = [b for b in gt_boxes if "car" in b.name or "truck" in b.name]
    for c in candidates:
        c_center = np.array(c["center_lidar_xyz"])
        c_yaw = float(c["yaw_lidar_rad"])
        best_dist = float("inf")
        best_gt = None
        for gt_box in vehicle_gt:
            d = float(np.linalg.norm(c_center - gt_box.center))
            if d < best_dist:
                best_dist = d
                best_gt = gt_box
        if best_dist > 5.0 or best_gt is None:
            continue
        gt_yaw = float(best_gt.orientation.yaw_pitch_roll[0])
        diff = float(((c_yaw - gt_yaw + np.pi / 2) % np.pi) - np.pi / 2)
        cid = c["candidate_id"]
        print(
            f"C{cid:02d}    {best_gt.token[:8]:10s} {best_dist:6.2f} "
            f"{np.degrees(c_yaw):9.1f} {np.degrees(gt_yaw):8.1f} {np.degrees(diff):8.1f} "
            f"{c['dimensions_lidar_lwh'][0]:7.3f} {best_gt.wlh[0]:7.3f} {best_gt.wlh[1]:7.3f}"
        )

    print()
    print("Интерпретация колонки 'diff' (нормировано на [-90, +90]):")
    print("  |diff| < 15 deg  → ориентация кандидата ВЕРНА")
    print("  |diff| ~= 90 deg → кандидат повёрнут на 90 относительно GT")
    print()
    print("Ключевой вопрос для барьеров:")
    print("  gt_L(wlh1) > gt_W(wlh0) → барьер вытянут вдоль wlh1 (length) — ось X бокса")
    print("  gt_L(wlh1) < gt_W(wlh0) → барьер вытянут вдоль wlh0 (width) — ось Y бокса")


if __name__ == "__main__":
    main()
