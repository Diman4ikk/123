from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import (
    BoxVisibility,
    points_in_box as nuscenes_points_in_box,
    view_points,
)



# Корневая папка датасета.
# Укажи тот же путь, который уже работает в evaluate_clusters.py.
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

# Параметры подготовки данных.
SCENE_NAME = "scene-0061"
MIN_LIDAR_POINTS = 5
MIN_CROP_SIZE = 2

OUTPUT_DIR = Path("outputs") / "fusion_sample"


def load_nuscenes() -> NuScenes:
    """Загружает nuScenes mini."""
    return NuScenes(
        version="v1.0-mini",
        dataroot=str(DATA_ROOT),
        verbose=False,
    )


def get_sample(
    nusc: NuScenes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Возвращает сцену и её первый keyframe."""
    scene = next(
        (
            item
            for item in nusc.scene
            if item["name"] == SCENE_NAME
        ),
        None,
    )

    if scene is None:
        raise RuntimeError(
            f"Сцена {SCENE_NAME!r} не найдена."
        )

    sample = nusc.get(
        "sample",
        scene["first_sample_token"],
    )

    return scene, sample


def project_box_to_image(
    box,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Проецирует углы 3D-бокса в систему координат изображения.

    box должен находиться в системе координат камеры.
    Возвращает обрезанные координаты (x1, y1, x2, y2).
    """
    corners = box.corners()

    # Точки за камерой нельзя корректно проецировать
    # обычным перспективным преобразованием.
    in_front = corners[2, :] > 1e-3

    if not np.any(in_front):
        return None

    projected = view_points(
        corners[:, in_front],
        intrinsic,
        normalize=True,
    )

    x_coordinates = projected[0, :]
    y_coordinates = projected[1, :]

    x1 = max(0, int(np.floor(x_coordinates.min())))
    y1 = max(0, int(np.floor(y_coordinates.min())))

    x2 = min(
        image_width,
        int(np.ceil(x_coordinates.max())),
    )
    y2 = min(
        image_height,
        int(np.ceil(y_coordinates.max())),
    )

    if x2 - x1 < MIN_CROP_SIZE:
        return None

    if y2 - y1 < MIN_CROP_SIZE:
        return None

    return x1, y1, x2, y2


def main() -> None:
    nusc = load_nuscenes()
    scene, sample = get_sample(nusc)

    camera_token = sample["data"]["CAM_FRONT"]
    lidar_token = sample["data"]["LIDAR_TOP"]

    # --------------------------------------------------
    # 1. Загружаем изображение камеры и её GT-боксы.
    # --------------------------------------------------
    image_path, camera_boxes, intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ANY,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Не удалось загрузить изображение: {image_path}"
        )

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------
    # 2. Загружаем облако LiDAR и его GT-боксы.
    # --------------------------------------------------
    lidar_path, lidar_boxes, _ = nusc.get_sample_data(
        lidar_token,
    )

    point_cloud = LidarPointCloud.from_file(lidar_path)

    # Форма массива: (3, N).
    lidar_points = point_cloud.points[:3, :]

    # Сопоставляем одни и те же аннотации между датчиками.
    lidar_boxes_by_token = {
        box.token: box
        for box in lidar_boxes
    }

    camera_timestamp = nusc.get(
        "sample_data",
        camera_token,
    )["timestamp"]

    lidar_timestamp = nusc.get(
        "sample_data",
        lidar_token,
    )["timestamp"]

    timestamp_delta_ms = (
        abs(camera_timestamp - lidar_timestamp) / 1000.0
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata: dict[str, Any] = {
        "scene": scene["name"],
        "sample_token": sample["token"],
        "camera": "CAM_FRONT",
        "lidar": "LIDAR_TOP",
        "camera_timestamp_us": camera_timestamp,
        "lidar_timestamp_us": lidar_timestamp,
        "timestamp_delta_ms": timestamp_delta_ms,
        "image_width": image_width,
        "image_height": image_height,
        "objects": [],
    }

    saved_count = 0
    skipped_count = 0

    # --------------------------------------------------
    # 3. Формируем пары «изображение + точки LiDAR».
    # --------------------------------------------------
    for camera_box in camera_boxes:
        annotation_token = camera_box.token

        lidar_box = lidar_boxes_by_token.get(
            annotation_token
        )

        if lidar_box is None:
            skipped_count += 1
            continue

        bbox = project_box_to_image(
            box=camera_box,
            intrinsic=intrinsic,
            image_width=image_width,
            image_height=image_height,
        )

        if bbox is None:
            skipped_count += 1
            continue

        x1, y1, x2, y2 = bbox

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            skipped_count += 1
            continue

        # points_in_box ожидает точки формы (3, N)
        # и бокс в той же системе координат, что и точки.
        points_mask = nuscenes_points_in_box(
            lidar_box,
            lidar_points,
        )

        object_points = (
            lidar_points[:, points_mask]
            .T
            .astype(np.float32)
        )

        if len(object_points) < MIN_LIDAR_POINTS:
            skipped_count += 1
            continue

        filename = f"{saved_count:04d}"

        image_filename = f"{filename}.jpg"
        points_filename = f"{filename}.npy"

        image_output_path = OUTPUT_DIR / image_filename
        points_output_path = OUTPUT_DIR / points_filename

        if not cv2.imwrite(
            str(image_output_path),
            crop,
        ):
            raise RuntimeError(
                f"Не удалось сохранить {image_output_path}"
            )

        np.save(
            points_output_path,
            object_points,
        )

        metadata["objects"].append(
            {
                "index": saved_count,
                "annotation_token": annotation_token,
                "category": camera_box.name,
                "image_file": image_filename,
                "lidar_file": points_filename,
                "bbox_xyxy": [x1, y1, x2, y2],
                "crop_width": x2 - x1,
                "crop_height": y2 - y1,
                "lidar_points_count": len(object_points),
            }
        )

        saved_count += 1

    # --------------------------------------------------
    # 4. Сохраняем метаданные.
    # --------------------------------------------------
    metadata_path = OUTPUT_DIR / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print("FUSION DATASET PREPARATION")
    print("=" * 60)
    print("Scene:", scene["name"])
    print("Sample token:", sample["token"])
    print("Camera image:", image_path)
    print("Image dimensions:", f"{image_width}x{image_height}")
    print("Visible camera boxes:", len(camera_boxes))
    print("LiDAR points:", lidar_points.shape[1])
    print(f"Sensor timestamp difference: {timestamp_delta_ms:.2f} ms")
    print("Saved objects:", saved_count)
    print("Skipped objects:", skipped_count)
    print("Output directory:", OUTPUT_DIR.resolve())
    print("Metadata:", metadata_path.resolve())


if __name__ == "__main__":
    main()