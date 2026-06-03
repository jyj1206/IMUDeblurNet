import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


K = np.array([
    [923.7181693, 0.0, 969.4457779],
    [0.0, 924.51235192, 532.9090534],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def read_image(path, scale):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if scale != 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def compute_rotation_matrix(theta):
    x, y, z = theta
    rx = np.array([[1, 0, 0], [0, np.cos(-x), -np.sin(-x)], [0, np.sin(-x), np.cos(-x)]])
    ry = np.array([[np.cos(-y), 0, np.sin(-y)], [0, 1, 0], [-np.sin(-y), 0, np.cos(-y)]])
    rz = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rx @ ry @ rz


def homography_flow(theta, height, width, scale):
    k = K.copy()
    k[0, :] *= scale
    k[1, :] *= scale
    h = k @ compute_rotation_matrix(theta) @ np.linalg.inv(k)

    ys = np.arange(height, dtype=np.float64)
    xs = np.arange(width, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys)
    points = np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)
    warped = points @ h.T
    warped = warped[..., :2] / warped[..., 2:3]
    return (warped - points[..., :2]).astype(np.float32)


def cosine_score(opt_flow, gyro_flow):
    mag_opt = np.linalg.norm(opt_flow, axis=2)
    mag_gyro = np.linalg.norm(gyro_flow, axis=2)
    mask = (mag_opt > np.percentile(mag_opt, 55)) & (mag_gyro > np.percentile(mag_gyro, 55))
    if mask.sum() < 100:
        return np.nan
    dot = (opt_flow[mask] * gyro_flow[mask]).sum(axis=1)
    denom = np.linalg.norm(opt_flow[mask], axis=1) * np.linalg.norm(gyro_flow[mask], axis=1) + 1e-8
    return float(np.median(dot / denom))


def load_rows(data_root, split, metadata_name, samples_per_scene, max_scenes, max_samples):
    with (data_root / split / metadata_name).open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    groups = {}
    for row in rows:
        groups.setdefault(row["scene_dir"], []).append(row)

    selected = []
    for scene_idx, scene_dir in enumerate(sorted(groups)):
        if max_scenes and scene_idx >= max_scenes:
            break
        scene_rows = groups[scene_dir]
        if len(scene_rows) <= samples_per_scene:
            selected.extend(scene_rows)
            continue
        positions = np.linspace(0, len(scene_rows) - 1, samples_per_scene).round().astype(int)
        selected.extend(scene_rows[int(pos)] for pos in positions)

    return selected[:max_samples] if max_samples else selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--metadata_name", default="metadata.csv")
    parser.add_argument("--samples_per_scene", type=int, default=3)
    parser.add_argument("--max_scenes", type=int, default=40)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--min_axis_theta", type=float, default=0.001)
    args = parser.parse_args()

    rows = load_rows(
        args.data_root,
        args.split,
        args.metadata_name,
        args.samples_per_scene,
        args.max_scenes,
        args.max_samples,
    )

    axis_names = ["x", "y", "z"]
    results = {axis: {"saved": [], "flipped": [], "theta": []} for axis in axis_names}
    scene_cache = {}
    used = 0

    for row in tqdm(rows, desc="verify per-axis direction"):
        scene_dir = row["scene_dir"]
        scene_root = args.data_root / args.split / scene_dir
        current_path = scene_root / row["target_sharp_path"]
        next_idx = int(row["center_frame_idx"]) + 1
        next_path = scene_root / "sharp" / f"frame_{next_idx:06d}.png"
        if not current_path.exists() or not next_path.exists():
            continue

        if scene_dir not in scene_cache:
            scene_cache[scene_dir] = (
                np.load(scene_root / "sensor_windows.npy", mmap_mode="r"),
                np.load(scene_root / "sensor_timestamps.npy", mmap_mode="r"),
            )
        sensor_windows, timestamps = scene_cache[scene_dir]
        sensor_idx = int(row["center_sensor_idx"])
        if sensor_idx + 1 >= len(sensor_windows):
            continue

        image_a = read_image(current_path, args.scale)
        image_b = read_image(next_path, args.scale)
        opt_flow = cv2.calcOpticalFlowFarneback(
            image_a,
            image_b,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        t0 = timestamps[sensor_idx, 3]
        t1 = timestamps[sensor_idx + 1, 3]
        dt = float(t1 - t0)
        if dt <= 0 or not np.isfinite(dt):
            continue

        gyro = 0.5 * (sensor_windows[sensor_idx, 3, 0:3] + sensor_windows[sensor_idx + 1, 3, 0:3])
        theta = gyro.astype(np.float64) * dt

        height, width = image_a.shape
        saved_flow = homography_flow(theta, height, width, args.scale)
        used += 1

        for axis_idx, axis_name in enumerate(axis_names):
            if abs(theta[axis_idx]) < args.min_axis_theta:
                continue
            flipped_theta = theta.copy()
            flipped_theta[axis_idx] *= -1
            flipped_flow = homography_flow(flipped_theta, height, width, args.scale)

            results[axis_name]["saved"].append(cosine_score(opt_flow, saved_flow))
            results[axis_name]["flipped"].append(cosine_score(opt_flow, flipped_flow))
            results[axis_name]["theta"].append(float(theta[axis_idx]))

    print(f"used_samples={used}")
    print("axis | valid | saved_median_cos | flipped_median_cos | saved_better_ratio | median_theta")
    for axis_name in axis_names:
        saved = np.array(results[axis_name]["saved"], dtype=np.float64)
        flipped = np.array(results[axis_name]["flipped"], dtype=np.float64)
        theta_values = np.array(results[axis_name]["theta"], dtype=np.float64)
        valid = np.isfinite(saved) & np.isfinite(flipped)
        if valid.sum() == 0:
            print(f"{axis_name:>4s} | 0 | nan | nan | nan | nan")
            continue
        saved_valid = saved[valid]
        flipped_valid = flipped[valid]
        theta_valid = theta_values[valid]
        print(
            f"{axis_name:>4s} | {valid.sum():5d} | "
            f"{np.median(saved_valid):+.4f} | {np.median(flipped_valid):+.4f} | "
            f"{np.mean(saved_valid > flipped_valid):.3f} | {np.median(theta_valid):+.6f}"
        )


if __name__ == "__main__":
    main()
