"""Диагностика: вывод исходных данных калибровки, поз и временных меток
для CAM_FRONT и LIDAR_TOP одного sample из nuScenes."""

from __future__ import annotations

import numpy as np
from nuscenes.nuscenes import NuScenes

DATAROOT = r"C:\Users\Dima\nuscenes\v1.0-mini"
VERSION = "v1.0-mini"
SAMPLE_TOKEN = "023c4df2d451409881d8e6ea82f14704"
SENSORS = ("CAM_FRONT", "LIDAR_TOP")


def main() -> None:
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    sample = nusc.get("sample", SAMPLE_TOKEN)

    # Записи sample_data, calibrated_sensor и ego_pose для каждого датчика
    records = {}
    for sensor in SENSORS:
        sd = nusc.get("sample_data", sample["data"][sensor])
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        records[sensor] = (sd, cs, ep)

    np.set_printoptions(precision=6, suppress=True)

    print(f"Sample token: {SAMPLE_TOKEN}")

    # 1. Внутренние параметры камеры (только CAM_FRONT)
    _, cam_cs, _ = records["CAM_FRONT"]
    K = np.array(cam_cs["camera_intrinsic"])
    print("\n=== Матрица внутренних параметров K для CAM_FRONT ===")
    print(K)

    # 2. calibrated_sensor: датчик -> ego (translation + кватернион)
    print("\n=== calibrated_sensor ===")
    for sensor in SENSORS:
        _, cs, _ = records[sensor]
        print(f"{sensor}:")
        print("  translation:", cs["translation"])
        print("  rotation (w, x, y, z):", cs["rotation"])

    # 3. ego_pose: ego -> global (translation + кватернион)
    print("\n=== ego_pose ===")
    for sensor in SENSORS:
        sd, _, ep = records[sensor]
        print(f"{sensor}: (токен ego_pose {sd['ego_pose_token']})")
        print("  translation:", ep["translation"])
        print("  rotation (w, x, y, z):", ep["rotation"])

    # 4. Временные метки (в nuScenes хранятся в микросекундах)
    print("\n=== Временные метки ===")
    ts = {}
    for sensor in SENSORS:
        sd, _, ep = records[sensor]
        ts[sensor] = sd["timestamp"]
        print(f"{sensor}: timestamp sample_data = {sd['timestamp']} мкс, "
              f"timestamp ego_pose = {ep['timestamp']} мкс")

    dt_ms = (ts["CAM_FRONT"] - ts["LIDAR_TOP"]) / 1000.0
    print(f"\nCAM_FRONT - LIDAR_TOP: {dt_ms:.3f} мс (модуль: {abs(dt_ms):.3f} мс)")


if __name__ == "__main__":
    main()