from __future__ import annotations

import json
from collections import Counter
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


# Корневая папка nuScenes mini.
# Здесь находятся samples/, sweeps/ и вложенная v1.0-mini/.
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

# Новая папка, чтобы не перезаписывать предыдущие версии.
OUTPUT_DIR = Path("outputs") / "fusion_dataset_v3"

# Ограничения для обучающих примеров.
MIN_LIDAR_POINTS = 10
MIN_CROP_WIDTH = 24
MIN_CROP_HEIGHT = 24

# Сцены для валидации.
# Train и val не должны содержать кадры одной сцены.
VALIDATION_SCENES = {
    "scene-0757",
    "scene-0916",
}


# Семь классов первого эксперимента.
CATEGORY_TO_LABEL = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.construction": "special_machinery",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
}


def load_nuscenes() -> NuScenes:
    """Загружает nuScenes mini."""
    return NuScenes(
        version="v1.0-mini",
        dataroot=str(DATA_ROOT),
        verbose=False,
    )


def split_scenes(
    nusc: NuScenes,
) -> tuple[set[str], set[str]]:
    """Делит набор данных по сценам."""
    all_scenes = {
        scene["name"]
        for scene in nusc.scene
    }

    missing_scenes = VALIDATION_SCENES - all_scenes

    if missing_scenes:
        raise RuntimeError(
            f"Сцены для валидации не найдены: {missing_scenes}"
        )

    train_scenes = all_scenes - VALIDATION_SCENES

    if not train_scenes:
        raise RuntimeError(
            "После разделения не осталось обучающих сцен."
        )

    return train_scenes, VALIDATION_SCENES.copy()


def project_box_to_image(
    box: Any,
    intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """
    Проецирует углы 3D-бокса в координаты изображения.

    Бокс должен находиться в системе координат камеры.
    Возвращает (x1, y1, x2, y2).
    """
    corners = box.corners()

    # Учитываем только углы перед камерой.
    in_front = corners[2, :] > 1e-3

    if not np.any(in_front):
        return None

    projected = view_points(
        corners[:, in_front],
        intrinsic,
        normalize=True,
    )

    x1 = max(
        0,
        int(np.floor(projected[0].min())),
    )
    y1 = max(
        0,
        int(np.floor(projected[1].min())),
    )
    x2 = min(
        image_width,
        int(np.ceil(projected[0].max())),
    )
    y2 = min(
        image_height,
        int(np.ceil(projected[1].max())),
    )

    if x2 - x1 < MIN_CROP_WIDTH:
        return None

    if y2 - y1 < MIN_CROP_HEIGHT:
        return None

    return x1, y1, x2, y2


def save_json(
    path: Path,
    data: Any,
) -> None:
    """Сохраняет данные в JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def transform_points_to_object_frame(
    points: np.ndarray,
    box: Any,
) -> np.ndarray:
    """
    Переводит точки из системы LiDAR в локальную систему объекта.

    points имеет форму (N, 3).
    Результат имеет форму (N, 3).
    """
    box_center = np.asarray(
        box.center,
        dtype=np.float32,
    )

    box_rotation = np.asarray(
        box.orientation.rotation_matrix,
        dtype=np.float32,
    )

    # Сначала переносим начало координат в центр объекта,
    # затем компенсируем ориентацию его 3D-бокса.
    local_points = (points - box_center) @ box_rotation

    return np.ascontiguousarray(
        local_points,
        dtype=np.float32,
    )


def main() -> None:
    nusc = load_nuscenes()

    train_scenes, val_scenes = split_scenes(nusc)

    scene_names_by_token = {
        scene["token"]: scene["name"]
        for scene in nusc.scene
    }

    # Создаём каталоги для изображений и облаков точек.
    for split in ("train", "val"):
        (OUTPUT_DIR / split / "images").mkdir(
            parents=True,
            exist_ok=True,
        )
        (OUTPUT_DIR / split / "points").mkdir(
            parents=True,
            exist_ok=True,
        )

    records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
    }

    skipped: Counter[str] = Counter()

    saved_by_class: dict[str, Counter[str]] = {
        "train": Counter(),
        "val": Counter(),
    }

    total_visible_boxes = 0
    total_samples = len(nusc.sample)

    print("Preparing nuScenes mini fusion dataset")
    print("=" * 60)
    print("Total keyframes:", total_samples)
    print("Train scenes:", sorted(train_scenes))
    print("Validation scenes:", sorted(val_scenes))
    print("Output directory:", OUTPUT_DIR.resolve())

    for sample_index, sample in enumerate(
        nusc.sample,
        start=1,
    ):
        scene_name = scene_names_by_token[
            sample["scene_token"]
        ]

        split = (
            "val"
            if scene_name in val_scenes
            else "train"
        )

        camera_token = sample["data"]["CAM_FRONT"]
        lidar_token = sample["data"]["LIDAR_TOP"]

        # --------------------------------------------------
        # 1. Изображение и аннотации в системе камеры.
        # --------------------------------------------------
        image_path, camera_boxes, intrinsic = (
            nusc.get_sample_data(
                camera_token,
                box_vis_level=BoxVisibility.ANY,
            )
        )

        image = cv2.imread(str(image_path))

        if image is None:
            skipped["image_read_error"] += 1
            continue

        image_height, image_width = image.shape[:2]

        # --------------------------------------------------
        # 2. Облако LiDAR и аннотации в системе LiDAR.
        # --------------------------------------------------
        lidar_path, lidar_boxes, _ = nusc.get_sample_data(
            lidar_token,
        )

        point_cloud = LidarPointCloud.from_file(
            lidar_path
        )

        # Форма исходного облака: (3, N).
        lidar_points = point_cloud.points[:3, :]

        # Сопоставляем одни и те же аннотации двух датчиков
        # по токену объекта.
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
            abs(camera_timestamp - lidar_timestamp)
            / 1000.0
        )

        total_visible_boxes += len(camera_boxes)

        # --------------------------------------------------
        # 3. Формируем пары «изображение + облако точек».
        # --------------------------------------------------
        for camera_box in camera_boxes:
            annotation_token = camera_box.token

            label = CATEGORY_TO_LABEL.get(
                camera_box.name
            )

            if label is None:
                skipped["category_not_selected"] += 1
                continue

            lidar_box = lidar_boxes_by_token.get(
                annotation_token
            )

            if lidar_box is None:
                skipped["lidar_box_not_found"] += 1
                continue

            # Получаем область объекта на изображении.
            bbox = project_box_to_image(
                box=camera_box,
                intrinsic=intrinsic,
                image_width=image_width,
                image_height=image_height,
            )

            if bbox is None:
                skipped["invalid_or_small_crop"] += 1
                continue

            x1, y1, x2, y2 = bbox

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                skipped["empty_crop"] += 1
                continue

            # Выбираем точки внутри эталонного 3D-бокса.
            # И точки, и lidar_box находятся в системе LiDAR.
            point_mask = nuscenes_points_in_box(
                lidar_box,
                lidar_points,
            )

            object_points = (
                lidar_points[:, point_mask]
                .T
                .astype(np.float32)
            )

            if len(object_points) < MIN_LIDAR_POINTS:
                skipped["too_few_lidar_points"] += 1
                continue

            # Переводим точки из координат LiDAR в локальные
            # координаты объекта. Сохраняем ориентацию бокса
            # относительно объекта, но не позицию в сцене.
            object_points = transform_points_to_object_frame(
                object_points,
                lidar_box,
            )

            # Токен sample вместе с токеном аннотации
            # обеспечивает уникальное имя примера.
            object_id = (
                f"{sample['token']}_{annotation_token}"
            )

            image_filename = f"{object_id}.jpg"
            points_filename = f"{object_id}.npy"

            split_dir = OUTPUT_DIR / split

            image_output = (
                split_dir / "images" / image_filename
            )

            points_output = (
                split_dir / "points" / points_filename
            )

            if not cv2.imwrite(
                str(image_output),
                crop,
            ):
                raise RuntimeError(
                    f"Не удалось сохранить изображение: {image_output}"
                )

            np.save(
                points_output,
                object_points,
                allow_pickle=False,
            )

            records[split].append(
                {
                    "object_id": object_id,
                    "scene": scene_name,
                    "sample_token": sample["token"],
                    "annotation_token": annotation_token,
                    "label": label,
                    "raw_category": camera_box.name,
                    "image_file": f"images/{image_filename}",
                    "points_file": f"points/{points_filename}",
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "crop_width": x2 - x1,
                    "crop_height": y2 - y1,
                    "lidar_points_count": len(object_points),
                    "points_coordinate_frame": "object_local_xyz",
                    "camera_timestamp_us": camera_timestamp,
                    "lidar_timestamp_us": lidar_timestamp,
                    "timestamp_delta_ms": round(
                        timestamp_delta_ms,
                        3,
                    ),
                }
            )

            saved_by_class[split][label] += 1

        if sample_index % 50 == 0:
            print(
                f"Processed {sample_index}/{total_samples} "
                "keyframes..."
            )

    # --------------------------------------------------
    # 4. Сохраняем индексы train и val.
    # --------------------------------------------------
    for split in ("train", "val"):
        save_json(
            OUTPUT_DIR / split / "metadata.json",
            records[split],
        )

    # --------------------------------------------------
    # 5. Сохраняем сводку датасета.
    # --------------------------------------------------
    summary = {
        "dataset": "nuScenes mini",
        "camera": "CAM_FRONT",
        "lidar": "LIDAR_TOP",
        "split_strategy": "scene-wise",
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "total_keyframes": total_samples,
        "visible_camera_boxes": total_visible_boxes,
        "point_coordinate_frame": "object_local_xyz",
        "saved_objects": {
            split: len(records[split])
            for split in ("train", "val")
        },
        "class_counts": {
            split: dict(saved_by_class[split])
            for split in ("train", "val")
        },
        "skipped_by_reason": dict(skipped),
        "minimum_lidar_points": MIN_LIDAR_POINTS,
        "minimum_crop_size": [
            MIN_CROP_WIDTH,
            MIN_CROP_HEIGHT,
        ],
    }

    save_json(
        OUTPUT_DIR / "summary.json",
        summary,
    )

    print("\n" + "=" * 60)
    print("FUSION DATASET SUMMARY")
    print("=" * 60)

    print("Visible camera boxes:", total_visible_boxes)
    print("Coordinate frame:", "object_local_xyz")

    for split in ("train", "val"):
        print(f"\n{split.upper()}")
        print("Saved objects:", len(records[split]))
        print("Class counts:")

        for label, count in sorted(
            saved_by_class[split].items()
        ):
            print(f"  {label}: {count}")

    print("\nSkipped objects:")

    for reason, count in sorted(skipped.items()):
        print(f"  {reason}: {count}")

    print("\nOutput directory:", OUTPUT_DIR.resolve())
    print(
        "Summary:",
        (OUTPUT_DIR / "summary.json").resolve(),
    )


if __name__ == "__main__":
    main()