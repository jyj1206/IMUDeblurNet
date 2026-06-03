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


def robust_cosine(flow_a, flow_b):
    mag_a = np.linalg.norm(flow_a, axis=2)
    mag_b = np.linalg.norm(flow_b, axis=2)
    mask = (mag_a > np.percentile(mag_a, 55)) & (mag_b > np.percentile(mag_b, 55))
    if mask.sum() < 100:
        return np.nan
    dot = (flow_a[mask] * flow_b[mask]).sum(axis=1)
    denom = np.linalg.norm(flow_a[mask], axis=1) * np.linalg.norm(flow_b[mask], axis=1) + 1e-8
    return float(np.median(dot / denom))


def robust_scale_corr(flow_a, flow_b):
    a = flow_a.reshape(-1, 2)
    b = flow_b.reshape(-1, 2)
    mag_a = np.linalg.norm(a, axis=1)
    mag_b = np.linalg.norm(b, axis=1)
    mask = (mag_a > np.percentile(mag_a, 55)) & (mag_b > np.percentile(mag_b, 55))
    if mask.sum() < 100:
        return np.nan
    av = a[mask].reshape(-1)
    bv = b[mask].reshape(-1)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv) + 1e-8
    return float((av @ bv) / denom)


def axis_candidates():
    candidates = {
        "saved_xyz": ((0, 1, 2), (1, 1, 1)),
        "flip_x": ((0, 1, 2), (-1, 1, 1)),
        "flip_y": ((0, 1, 2), (1, -1, 1)),
        "flip_z": ((0, 1, 2), (1, 1, -1)),
        "flip_xy": ((0, 1, 2), (-1, -1, 1)),
        "flip_xz": ((0, 1, 2), (-1, 1, -1)),
        "flip_yz": ((0, 1, 2), (1, -1, -1)),
        "flip_xyz": ((0, 1, 2), (-1, -1, -1)),
        "swap_xy": ((1, 0, 2), (1, 1, 1)),
        "raw_y_negx_z_guess": ((1, 0, 2), (1, -1, 1)),
    }
    return candidates


def load_rows(data_root, split, metadata_name, max_samples, samples_per_scene, max_scenes):
    with (data_root / split / metadata_name).open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if samples_per_scene:
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
        rows = selected
    return rows[:max_samples] if max_samples else rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--metadata_name", default="metadata.csv")
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--samples_per_scene", type=int, default=0)
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.25)
    args = parser.parse_args()

    rows = load_rows(
        args.data_root,
        args.split,
        args.metadata_name,
        args.max_samples,
        args.samples_per_scene,
        args.max_scenes,
    )
    candidates = axis_candidates()
    scores = {name: {"cos": [], "corr": []} for name in candidates}
    scene_cache = {}
    used = 0

    for row in tqdm(rows, desc="verify gyro axes"):
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

        for name, (perm, signs) in candidates.items():
            theta = gyro[list(perm)] * np.array(signs, dtype=np.float64) * dt
            gyro_flow = homography_flow(theta, image_a.shape[0], image_a.shape[1], args.scale)
            scores[name]["cos"].append(robust_cosine(opt_flow, gyro_flow))
            scores[name]["corr"].append(robust_scale_corr(opt_flow, gyro_flow))
        used += 1

    summary = []
    for name, value in scores.items():
        cos = np.array(value["cos"], dtype=np.float64)
        corr = np.array(value["corr"], dtype=np.float64)
        summary.append((
            float(np.nanmedian(cos)),
            float(np.nanmedian(corr)),
            name,
            int(np.isfinite(cos).sum()),
        ))

    print(f"used_samples={used}")
    print("rank | candidate | median_cos | median_corr | valid")
    for rank, (cos, corr, name, valid) in enumerate(sorted(summary, reverse=True), 1):
        print(f"{rank:02d} | {name:20s} | {cos:+.4f} | {corr:+.4f} | {valid}")


if __name__ == "__main__":
    main()
