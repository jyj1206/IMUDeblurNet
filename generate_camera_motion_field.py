import argparse
import csv
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


WORKER_ARGS = None
WORKER_CENTER_VECTOR_CACHE = None
WORKER_SCENE_CACHE = None


def compute_rotation_matrix(ang_vel_x, ang_vel_y, ang_vel_z):
    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(-ang_vel_x), -np.sin(-ang_vel_x)],
        [0, np.sin(-ang_vel_x), np.cos(-ang_vel_x)],
    ])
    r_y = np.array([
        [np.cos(-ang_vel_y), 0, np.sin(-ang_vel_y)],
        [0, 1, 0],
        [-np.sin(-ang_vel_y), 0, np.cos(-ang_vel_y)],
    ])
    r_z = np.array([
        [np.cos(ang_vel_z), np.sin(ang_vel_z), 0],
        [-np.sin(ang_vel_z), np.cos(ang_vel_z), 0],
        [0, 0, 1],
    ])
    return r_x @ r_y @ r_z


def compute_homography(r):
    k = np.array([
        [923.7181693, 0.0, 969.4457779],
        [0.0, 924.51235192, 532.9090534],
        [0.0, 0.0, 1.0],
    ])
    return k @ r @ np.linalg.inv(k)


def read_image_shape(image_path):
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image.shape[:2]


def build_center_vectors(height, width, downsample):
    ys = np.arange(0, height, downsample, dtype=np.float64)
    xs = np.arange(0, width, downsample, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, ys)
    return np.stack([x_grid, y_grid, np.ones_like(x_grid)], axis=-1)


def compute_interval_rotations(gyro_window, timestamp_window, default_dt):
    gyro_window = gyro_window.astype(np.float64)
    timestamp_window = timestamp_window.astype(np.float64)

    dt = np.diff(timestamp_window)
    if np.nanmedian(dt) > 1e6:
        dt = dt * 1e-9
    if len(dt) != 6 or not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        dt = np.full(6, default_dt, dtype=np.float64)

    r_list = []
    for idx in range(6):
        theta = 0.5 * (gyro_window[idx] + gyro_window[idx + 1]) * dt[idx]
        r_list.append(compute_rotation_matrix(theta[0], theta[1], theta[2]))
    return r_list


def make_camera_motion_field(gyro_window, timestamp_window, center_vectors, default_dt):
    r_list = compute_interval_rotations(gyro_window, timestamp_window, default_dt)

    r = np.eye(3)
    h_pro_list = []
    for idx in range(len(r_list) // 2, len(r_list)):
        r = r_list[idx] @ r
        h_pro_list.append(compute_homography(r))

    r = np.eye(3)
    h_pre_list = []
    for idx in range((len(r_list) // 2) - 1, -1, -1):
        r = r @ r_list[idx]
        h_pre_list.append(np.linalg.inv(compute_homography(r)))

    cmf_pro = None
    for h_pro in h_pro_list:
        end_vectors = np.einsum("ij,klj->kli", h_pro, center_vectors)
        end_vectors = end_vectors / end_vectors[:, :, -1, np.newaxis]
        vector = end_vectors[:, :, :2] - center_vectors[:, :, :2]
        cmf_pro = vector.copy() if cmf_pro is None else np.concatenate((cmf_pro, vector), axis=2)

    cmf_pre = None
    for h_pre in h_pre_list:
        initial_vectors = np.einsum("ij,klj->kli", h_pre, center_vectors)
        initial_vectors = initial_vectors / initial_vectors[:, :, -1, np.newaxis]
        vector = initial_vectors[:, :, :2] - center_vectors[:, :, :2]
        cmf_pre = vector.copy() if cmf_pre is None else np.concatenate((vector, cmf_pre), axis=2)

    cmf = np.concatenate((cmf_pre, cmf_pro), axis=2)

    for idx in range(5, 3, -1):
        cmf[:, :, idx * 2 : idx * 2 + 2] -= cmf[:, :, idx * 2 - 2 : idx * 2]

    for idx in range(0, 2):
        cmf[:, :, idx * 2 : idx * 2 + 2] -= cmf[:, :, idx * 2 + 2 : idx * 2 + 4]

    cmf[:, :, 0:6] = -cmf[:, :, 0:6]
    return cmf


def output_path(save_root_dir, mode, scene_dir, blur_path, save_format):
    return save_root_dir / mode / "camera_motion_field" / scene_dir / f"{Path(blur_path).stem}.{save_format}"


def row_blur_path(row):
    if row.get("blur_path"):
        return row["blur_path"]
    return row["center_blur_path"]


def row_sensor_idx(row):
    return int(row.get("sensor_idx") or row.get("center_sensor_idx") or 0)


def save_motion_field(path, motion_field, dtype, save_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    motion_field = motion_field.astype(dtype)
    if save_format == "npy":
        np.save(path, motion_field)
        other_path = path.with_suffix(".npz")
    else:
        np.savez_compressed(path, motion_field=motion_field)
        other_path = path.with_suffix(".npy")

    if other_path.exists():
        other_path.unlink()


def generate_camera_motion_field(row, mode, data_root, save_root_dir, center_vector_cache, args, scene_cache=None):
    split_root = data_root / mode
    scene_dir = row["scene_dir"]
    scene_root = split_root / scene_dir
    blur_path = row_blur_path(row)
    save_file = output_path(save_root_dir, mode, scene_dir, blur_path, args.save_format)

    if save_file.exists() and not args.overwrite:
        return False

    blur_file = scene_root / blur_path
    sensor_file = scene_root / "sensor_windows.npy"
    timestamp_file = scene_root / "sensor_timestamps.npy"
    missing_files = [path for path in (scene_root, blur_file, sensor_file, timestamp_file) if not path.exists()]
    if missing_files:
        if args.strict:
            raise FileNotFoundError(f"Missing files for {mode}/{scene_dir}: {missing_files}")
        return False

    height, width = read_image_shape(blur_file)
    cache_key = (height, width, args.downsample)
    if cache_key not in center_vector_cache:
        center_vector_cache[cache_key] = build_center_vectors(height, width, args.downsample)

    sensor_idx = row_sensor_idx(row)
    if scene_cache is not None:
        if scene_dir not in scene_cache:
            scene_cache[scene_dir] = (
                np.load(sensor_file, mmap_mode="r"),
                np.load(timestamp_file, mmap_mode="r"),
            )
        sensor_windows, sensor_timestamps = scene_cache[scene_dir]
    else:
        sensor_windows = np.load(sensor_file, mmap_mode="r")
        sensor_timestamps = np.load(timestamp_file, mmap_mode="r")
    gyro_window = np.asarray(sensor_windows[sensor_idx, :, 0:3])
    timestamp_window = np.asarray(sensor_timestamps[sensor_idx])

    cmf = make_camera_motion_field(
        gyro_window=gyro_window,
        timestamp_window=timestamp_window,
        center_vectors=center_vector_cache[cache_key],
        default_dt=args.default_dt,
    )

    save_motion_field(save_file, cmf, args.dtype, args.save_format)
    return True


def init_worker(worker_args):
    global WORKER_ARGS
    global WORKER_CENTER_VECTOR_CACHE
    global WORKER_SCENE_CACHE

    WORKER_ARGS = argparse.Namespace(**worker_args)
    WORKER_ARGS.data_root = Path(WORKER_ARGS.data_root)
    WORKER_ARGS.save_root_dir = Path(WORKER_ARGS.save_root_dir)
    WORKER_CENTER_VECTOR_CACHE = {}
    WORKER_SCENE_CACHE = {}


def generate_row_worker(row):
    return int(
        generate_camera_motion_field(
            row=row,
            mode=WORKER_ARGS.mode,
            data_root=WORKER_ARGS.data_root,
            save_root_dir=WORKER_ARGS.save_root_dir,
            center_vector_cache=WORKER_CENTER_VECTOR_CACHE,
            args=WORKER_ARGS,
            scene_cache=WORKER_SCENE_CACHE,
        )
    )


def generate_mode(args, mode):
    metadata_file = args.data_root / mode / args.metadata_name
    if not metadata_file.exists():
        print(f"skip missing split: {args.data_root / mode}")
        return

    with metadata_file.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if args.max_samples:
        rows = rows[:args.max_samples]

    save_root_dir = args.save_root_dir or args.data_root
    save_count = 0
    if args.num_workers > 1:
        worker_args = vars(args).copy()
        worker_args["mode"] = mode
        worker_args["data_root"] = str(args.data_root)
        worker_args["save_root_dir"] = str(save_root_dir)
        with mp.Pool(processes=args.num_workers, initializer=init_worker, initargs=(worker_args,)) as pool:
            iterator = pool.imap_unordered(generate_row_worker, rows, chunksize=args.chunksize)
            for saved in tqdm(iterator, total=len(rows), desc=f"{mode} camera_motion_field"):
                save_count += saved
    else:
        center_vector_cache = {}
        scene_cache = {}
        for row in tqdm(rows, desc=f"{mode} camera_motion_field"):
            saved = generate_camera_motion_field(
                row=row,
                mode=mode,
                data_root=args.data_root,
                save_root_dir=save_root_dir,
                center_vector_cache=center_vector_cache,
                args=args,
                scene_cache=scene_cache,
            )
            save_count += int(saved)

    print(f"{mode}: saved {save_count} files under {save_root_dir / mode / 'camera_motion_field'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--data_root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--save_root_dir", type=Path)
    parser.add_argument("--metadata_name", default="metadata.csv")
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--default_dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--save_format", choices=["npy", "npz"], default="npy")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    modes = ["train", "val", "test"] if args.mode == "all" else [args.mode]
    for mode in modes:
        generate_mode(args, mode)


if __name__ == "__main__":
    mp.freeze_support()
    main()
