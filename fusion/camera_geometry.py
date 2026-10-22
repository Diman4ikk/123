"""Camera-LiDAR geometry utilities for the nuScenes fusion project."""

from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion


DATAROOT = r"C:\Users\Dima\nuscenes\v1.0-mini"
VERSION = "v1.0-mini"
SAMPLE_TOKEN = "023c4df2d451409881d8e6ea82f14704"


def sensor_to_global_matrix(
    nusc: NuScenes,
    sample_data: dict,
) -> np.ndarray:
    """Return a 4x4 matrix transforming sensor coordinates to global coordinates."""
    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )
    ego_pose = nusc.get(
        "ego_pose",
        sample_data["ego_pose_token"],
    )

    # Sensor coordinates -> ego coordinates.
    T_ego_from_sensor = transform_matrix(
        calibrated_sensor["translation"],
        Quaternion(calibrated_sensor["rotation"]),
        inverse=False,
    )

    # Ego coordinates -> global coordinates.
    T_global_from_ego = transform_matrix(
        ego_pose["translation"],
        Quaternion(ego_pose["rotation"]),
        inverse=False,
    )

    return T_global_from_ego @ T_ego_from_sensor


def transform_lidar_points_to_camera(
    points_lidar: np.ndarray,
    T_camera_from_lidar: np.ndarray,
) -> np.ndarray:
    """Transform LiDAR points from N x 3 or N x 5 to camera coordinates N x 3."""
    points_lidar = np.asarray(points_lidar)

    if points_lidar.ndim != 2 or points_lidar.shape[1] < 3:
        raise ValueError(
            f"Expected points shaped [N, 3] or [N, 5], "
            f"got {points_lidar.shape}"
        )

    if T_camera_from_lidar.shape != (4, 4):
        raise ValueError("T_camera_from_lidar must have shape (4, 4)")

    # Keep only x, y, z. Ignore intensity and ring if present.
    xyz = points_lidar[:, :3].astype(np.float64)

    # Convert to homogeneous coordinates: [x, y, z] -> [x, y, z, 1].
    ones = np.ones((xyz.shape[0], 1), dtype=np.float64)
    xyz_homogeneous = np.concatenate([xyz, ones], axis=1)

    # Row-wise points multiplied by the transposed transformation matrix.
    points_camera_homogeneous = (
        T_camera_from_lidar @ xyz_homogeneous.T
    ).T

    return points_camera_homogeneous[:, :3]


def project_lidar_points(
    points_lidar: np.ndarray,
    T_camera_from_lidar: np.ndarray,
    K: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project LiDAR points to image pixels.

    Returns:
        pixels: float array [M, 2] containing (u, v) image coordinates.
        mask: bool array [N], True for original points that project inside
              the image and have positive camera depth.
    """
    points_lidar = np.asarray(points_lidar)

    if points_lidar.ndim != 2 or points_lidar.shape[1] < 3:
        raise ValueError(
            f"Expected points shaped [N, 3] or [N, 5], "
            f"got {points_lidar.shape}"
        )

    if K.shape != (3, 3):
        raise ValueError("K must have shape (3, 3)")

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive")

    points_camera = transform_lidar_points_to_camera(
        points_lidar,
        T_camera_from_lidar,
    )

    x_camera = points_camera[:, 0]
    y_camera = points_camera[:, 1]
    z_camera = points_camera[:, 2]

    # A point must have finite coordinates and lie in front of the camera.
    in_front = (
        np.isfinite(points_camera).all(axis=1)
        & (z_camera > 0)
    )

    u = np.full(len(points_lidar), np.nan, dtype=np.float64)
    v = np.full(len(points_lidar), np.nan, dtype=np.float64)

    # nuScenes camera intrinsic matrix: zero skew, standard pinhole model.
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u[in_front] = (
        fx * x_camera[in_front] / z_camera[in_front] + cx
    )
    v[in_front] = (
        fy * y_camera[in_front] / z_camera[in_front] + cy
    )

    # Keep only points inside the image boundaries.
    mask = (
        in_front
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )

    pixels = np.column_stack((u[mask], v[mask]))

    return pixels, mask


def main() -> None:
    nusc = NuScenes(
        version=VERSION,
        dataroot=DATAROOT,
        verbose=False,
    )

    sample = nusc.get("sample", SAMPLE_TOKEN)

    # Get the camera and LiDAR sample_data records.
    camera_data = nusc.get(
        "sample_data",
        sample["data"]["CAM_FRONT"],
    )
    lidar_data = nusc.get(
        "sample_data",
        sample["data"]["LIDAR_TOP"],
    )

    # Calculate sensor-to-global transformations.
    T_global_from_camera = sensor_to_global_matrix(
        nusc,
        camera_data,
    )
    T_global_from_lidar = sensor_to_global_matrix(
        nusc,
        lidar_data,
    )

    # LiDAR coordinates -> camera coordinates.
    T_camera_from_lidar = (
        np.linalg.inv(T_global_from_camera)
        @ T_global_from_lidar
    )

    print(f"Sample token: {SAMPLE_TOKEN}")

    print("\n=== T_global_from_camera ===")
    print(T_global_from_camera)

    print("\n=== T_global_from_lidar ===")
    print(T_global_from_lidar)

    print("\n=== T_camera_from_lidar ===")
    print(T_camera_from_lidar)

    print("\nTransformation shape:", T_camera_from_lidar.shape)
    print(
        "Transformation finite:",
        bool(np.isfinite(T_camera_from_lidar).all()),
    )

    # Read the camera intrinsic matrix K.
    camera_calibration = nusc.get(
        "calibrated_sensor",
        camera_data["calibrated_sensor_token"],
    )
    K = np.asarray(
        camera_calibration["camera_intrinsic"],
        dtype=np.float64,
    )

    # Load the raw LiDAR point cloud: x, y, z, intensity, ring.
    lidar_path = nusc.get_sample_data_path(
        sample["data"]["LIDAR_TOP"]
    )
    raw_points = np.fromfile(lidar_path, dtype=np.float32)

    if raw_points.size % 5 != 0:
        raise ValueError(
            f"LiDAR file contains {raw_points.size} values; "
            "the count is not divisible by 5"
        )

    points_lidar = raw_points.reshape(-1, 5)

    # Get the actual image dimensions.
    camera_path = nusc.get_sample_data_path(
        sample["data"]["CAM_FRONT"]
    )
    with Image.open(camera_path) as image:
        image_width, image_height = image.size

    # Count points with positive depth before applying image boundaries.
    points_camera = transform_lidar_points_to_camera(
        points_lidar,
        T_camera_from_lidar,
    )
    in_front = (
        np.isfinite(points_camera).all(axis=1)
        & (points_camera[:, 2] > 0)
    )

    # Project the points into the camera image.
    pixels, in_image = project_lidar_points(
        points_lidar=points_lidar,
        T_camera_from_lidar=T_camera_from_lidar,
        K=K,
        image_width=image_width,
        image_height=image_height,
    )
        # Save an image with projected LiDAR points.
    output_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "camera_geometry"
        / "cam_front_lidar_projection.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(camera_path) as image:
        overlay = image.convert("RGB")

    draw = ImageDraw.Draw(overlay)

    for u, v in pixels:
        x = int(round(u))
        y = int(round(v))

        draw.ellipse(
            (x - 2, y - 2, x + 2, y + 2),
            fill=(0, 255, 0),
        )

    overlay.save(output_path)
    print("Projection overlay saved:", output_path)
    print("\n=== Projection results ===")
    print("Total LiDAR points:", len(points_lidar))
    print("Image dimensions:", f"{image_width} x {image_height}")
    print("Points in front of camera:", int(in_front.sum()))
    print("Points inside image:", int(in_image.sum()))
    print("Pixel array shape:", pixels.shape)
    print("Pixel coordinates finite:", bool(np.isfinite(pixels).all()))

    print("\n=== Timestamps ===")
    camera_timestamp = camera_data["timestamp"]
    lidar_timestamp = lidar_data["timestamp"]
    delta_ms = (camera_timestamp - lidar_timestamp) / 1000.0

    print(f"CAM_FRONT: {camera_timestamp} microseconds")
    print(f"LIDAR_TOP: {lidar_timestamp} microseconds")
    print(f"CAM_FRONT - LIDAR_TOP: {delta_ms:.3f} ms")


if __name__ == "__main__":
    main()