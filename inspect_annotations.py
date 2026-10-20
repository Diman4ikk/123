from pathlib import Path

from nuscenes.nuscenes import NuScenes

DATA_ROOT = Path(r"C:\Users\Dima\nuscenes\v1.0-mini")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

scene = nusc.scene[0]
sample = nusc.get("sample", scene["first_sample_token"])

lidar_token = sample["data"]["LIDAR_TOP"]

# Получаем аннотации, преобразованные в координаты LiDAR.
_, boxes, _ = nusc.get_sample_data(lidar_token)

print("Scene:", scene["name"])
print("Sample:", sample["token"])
print("\nGROUND TRUTH IN LIDAR COORDINATES")
print("-" * 70)

for index, box in enumerate(boxes):
    center = box.center
    width, length, height = box.wlh

    print(
        f"Object {index + 1}: {box.name}\n"
        f"  Center XYZ: "
        f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n"
        f"  Size W/L/H: "
        f"({width:.2f}, {length:.2f}, {height:.2f})\n"
    )

print("Total annotated objects:", len(boxes))