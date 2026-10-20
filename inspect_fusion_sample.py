import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path("outputs") / "fusion_sample"
SHEET_PATH = DATA_DIR / "contact_sheet.jpg"

TILE_WIDTH = 420
TILE_HEIGHT = 280
COLUMNS = 3


def main() -> None:
    metadata_path = DATA_DIR / "metadata.json"

    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    objects = metadata["objects"]

    if not objects:
        raise RuntimeError("В датасете нет сохранённых объектов.")

    rows = (len(objects) + COLUMNS - 1) // COLUMNS

    sheet = np.full(
        (rows * TILE_HEIGHT, COLUMNS * TILE_WIDTH, 3),
        35,
        dtype=np.uint8,
    )

    categories = Counter()
    errors = []

    for index, item in enumerate(objects):
        image_path = DATA_DIR / item["image_file"]
        points_path = DATA_DIR / item["lidar_file"]

        image = cv2.imread(str(image_path))

        if image is None:
            errors.append(f"Не удалось прочитать {image_path.name}")
            continue

        if not points_path.exists():
            errors.append(f"Не найден файл {points_path.name}")
            continue

        points = np.load(points_path, allow_pickle=False)

        if points.ndim != 2 or points.shape[1] != 3:
            errors.append(
                f"{points_path.name}: неверная форма {points.shape}"
            )
            continue

        if not np.isfinite(points).all():
            errors.append(
                f"{points_path.name}: обнаружены некорректные координаты"
            )
            continue

        if len(points) != item["lidar_points_count"]:
            errors.append(
                f"{points_path.name}: число точек не совпадает с metadata"
            )
            continue

        category = item["category"]
        categories[category] += 1

        # Подгоняем изображение под область предпросмотра,
        # сохраняя исходные пропорции.
        preview_width = TILE_WIDTH - 20
        preview_height = TILE_HEIGHT - 65

        scale = min(
            preview_width / image.shape[1],
            preview_height / image.shape[0],
        )

        width = max(1, int(image.shape[1] * scale))
        height = max(1, int(image.shape[0] * scale))

        preview = cv2.resize(image, (width, height))

        row = index // COLUMNS
        column = index % COLUMNS

        x = column * TILE_WIDTH
        y = row * TILE_HEIGHT

        cv2.putText(
            sheet,
            f"{index:02d} | {category}",
            (x + 10, y + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            f"LiDAR points: {len(points)}",
            (x + 10, y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )

        preview_x = x + (TILE_WIDTH - width) // 2
        preview_y = y + 55

        sheet[
            preview_y:preview_y + height,
            preview_x:preview_x + width,
        ] = preview

    if not cv2.imwrite(str(SHEET_PATH), sheet):
        raise RuntimeError("Не удалось сохранить контактный лист.")

    print("Objects:", len(objects))
    print("\nCategories:")

    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")

    print("\nValidation errors:", len(errors))

    for error in errors:
        print(" ", error)

    print("\nContact sheet:", SHEET_PATH.resolve())


if __name__ == "__main__":
    main()