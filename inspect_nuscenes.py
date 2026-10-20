from nuscenes.nuscenes import NuScenes

DATA_ROOT = r"C:\Users\Dima\nuscenes\v1.0-mini"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATA_ROOT,
    verbose=True,
)

scene = nusc.scene[0]

print("\nSCENE")
print("name:", scene["name"])
print("description:", scene["description"])

sample = nusc.get("sample", scene["first_sample_token"])

print("\nSAMPLE")
print("token:", sample["token"])

print("\nSENSORS")

for channel, token in sample["data"].items():
    sample_data = nusc.get("sample_data", token)

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )

    sensor = nusc.get(
        "sensor",
        calibrated_sensor["sensor_token"],
    )

    print(f"\n{channel}")
    print("  filename:", sample_data["filename"])
    print("  modality:", sensor["modality"])