from pathlib import Path

import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility


# Скопируй сюда DATA_ROOT из своего работающего evaluate_clusters.py.
DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

OUTPUT_DIR = Path("outputs")


def main() -> None:
    nusc = NuScenes(
        version="v1.0-mini",
        dataroot=str(DATA_ROOT),
        verbose=False,
    )

    # Берём ту же сцену, которую используем при оценке кластеров.
    scene = next(
        item
        for item in nusc.scene
        if item["name"] == "scene-0061"
    )

    sample = nusc.get(
        "sample",
        scene["first_sample_token"],
    )

    camera_token = sample["data"]["CAM_FRONT"]

    # Получаем изображение, боксы в системе координат камеры
    # и внутреннюю матрицу калибровки камеры.
    image_path, boxes, camera_intrinsic = nusc.get_sample_data(
        camera_token,
        box_vis_level=BoxVisibility.ANY,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Не удалось загрузить изображение: {image_path}"
        )

    # Рисуем эталонные 3D-боксы непосредственно на изображении.
    green = (0, 255, 0)

    for box in boxes:
        box.render_cv2(
            image,
            view=camera_intrinsic,
            normalize=True,
            colors=(green, green, green),
            linewidth=2,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / "cam_front_gt.jpg"

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Не удалось сохранить изображение: {output_path}"
        )

    print("Scene:", scene["name"])
    print("Camera image:", image_path)
    print("Visible GT boxes:", len(boxes))
    print("Saved:", output_path.resolve())


if __name__ == "__main__":
    main()