from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .image_dataset_common import (
    find_metadata,
    load_image,
    read_csv,
    resolve_split_name,
    scene_root,
)


def _as_hw(image_size):
    if image_size is None:
        return None
    if isinstance(image_size, int):
        return image_size, image_size
    return int(image_size[0]), int(image_size[1])


def _resize_image(image, image_size):
    hw = _as_hw(image_size)
    if hw is None:
        return image
    return F.interpolate(
        image.unsqueeze(0),
        size=hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _normalize_image(image, mean, std):
    if mean is None or std is None:
        return image
    mean = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean) / std


def _default_timestamp_window(num_vectors, default_dt):
    return np.arange(int(num_vectors), dtype=np.float32) * float(default_dt)


class Stage1Stage2Dataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        metadata_name="metadata.csv",
        stage1_image_size=(224, 320),
        stage1_mean=None,
        stage1_std=None,
        num_vectors=7,
        vector_start=0,
        vector_dim=3,
        load_target_gyro=False,
        default_dt=1.0 / 240.0,
        motion_field_root=None,
        motion_field_dir="camera_motion_field",
        motion_field_ext="npy",
    ):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.split_root = self.dataset_root / self.split
        self.stage1_image_size = stage1_image_size
        self.stage1_mean = stage1_mean
        self.stage1_std = stage1_std
        self.num_vectors = int(num_vectors)
        self.vector_start = int(vector_start)
        self.vector_dim = int(vector_dim)
        self.load_target_gyro = bool(load_target_gyro)
        self.default_dt = float(default_dt)
        self.motion_field_root = Path(motion_field_root) if motion_field_root else self.dataset_root
        self.motion_field_dir = motion_field_dir
        self.motion_field_ext = motion_field_ext
        self.sensor_cache = {}
        self.timestamp_cache = {}

        self.metadata_path = find_metadata(
            self.split_root,
            metadata_name,
            ["paired_metadata.csv", "metadata.csv"],
        )
        self.fieldnames, self.rows = read_csv(self.metadata_path)
        self.layout = self._detect_layout()

    def _detect_layout(self):
        if not self.rows:
            return "paired"

        columns = set(self.rows[0].keys())
        if {"blur_path", "sharp_path"}.issubset(columns):
            return "paired"
        if {"center_blur_path", "target_sharp_path"}.issubset(columns):
            return "triplet_center_fallback"

        raise ValueError(
            f"{self.metadata_path} must contain blur_path/sharp_path or "
            "center_blur_path/target_sharp_path."
        )

    def _image_paths(self, row):
        root = scene_root(self.split_root, row)
        if self.layout == "paired":
            return root / row["blur_path"], root / row["sharp_path"]
        return root / row["center_blur_path"], root / row["target_sharp_path"]

    def _motion_field_path(self, row):
        scene_dir = row.get("scene_dir", "")
        blur_path = row["blur_path"] if self.layout == "paired" else row["center_blur_path"]
        motion_name = f"{Path(blur_path).stem}.{self.motion_field_ext}"
        return self.motion_field_root / self.split / self.motion_field_dir / scene_dir / motion_name

    def _sensor_idx(self, row):
        return int(row.get("sensor_idx") or row.get("center_sensor_idx") or 0)

    def _load_sensor_windows(self, row):
        scene_dir = row.get("scene_dir", "")
        if not scene_dir:
            raise FileNotFoundError("gyro target loading needs scene_dir in metadata.")
        if scene_dir not in self.sensor_cache:
            path = self.split_root / scene_dir / "sensor_windows.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing sensor_windows.npy: {path}")
            self.sensor_cache[scene_dir] = np.load(path, mmap_mode="r")
        return self.sensor_cache[scene_dir]

    def _load_target_gyro(self, row):
        sensor_windows = self._load_sensor_windows(row)
        sensor_idx = self._sensor_idx(row)
        end = self.vector_start + self.vector_dim
        target_gyro = np.asarray(
            sensor_windows[sensor_idx, : self.num_vectors, self.vector_start:end],
            dtype=np.float32,
        )
        if target_gyro.shape != (self.num_vectors, self.vector_dim):
            raise ValueError(
                f"gyro target shape must be {(self.num_vectors, self.vector_dim)}, "
                f"got {target_gyro.shape}"
            )
        return torch.from_numpy(np.array(target_gyro, dtype=np.float32, copy=True))

    def _load_timestamp_window(self, row):
        scene_dir = row.get("scene_dir", "")
        if not scene_dir:
            return torch.from_numpy(_default_timestamp_window(self.num_vectors, self.default_dt))

        if scene_dir not in self.timestamp_cache:
            path = self.split_root / scene_dir / "sensor_timestamps.npy"
            self.timestamp_cache[scene_dir] = (
                np.load(path, mmap_mode="r") if path.exists() else None
            )

        timestamps = self.timestamp_cache[scene_dir]
        if timestamps is None:
            window = _default_timestamp_window(self.num_vectors, self.default_dt)
        else:
            sensor_idx = self._sensor_idx(row)
            window = np.asarray(timestamps[sensor_idx, : self.num_vectors], dtype=np.float32)
            if window.shape[0] != self.num_vectors:
                window = _default_timestamp_window(self.num_vectors, self.default_dt)
        return torch.from_numpy(np.array(window, dtype=np.float32, copy=True))

    def _sample_meta(self, index, row, lq_path, gt_path):
        return {
            "index": int(index),
            "type": row.get("type", "unknown"),
            "scene_dir": row.get("scene_dir", ""),
            "stem": Path(lq_path).stem,
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
            "motion_field_path": str(self._motion_field_path(row)),
            "sensor_idx": self._sensor_idx(row),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lq_path, gt_path = self._image_paths(row)
        blur = load_image(lq_path)
        sharp = load_image(gt_path)
        stage1_image = _normalize_image(
            _resize_image(blur, self.stage1_image_size),
            self.stage1_mean,
            self.stage1_std,
        )

        sample = {
            "stage1_image": stage1_image,
            "lq": blur,
            "gt": sharp,
            "timestamp_window": self._load_timestamp_window(row),
            "meta": self._sample_meta(index, row, lq_path, gt_path),
        }
        if self.load_target_gyro:
            sample["gyro"] = self._load_target_gyro(row)
        return sample


def build_stage1_stage2_dataset(
    stage1_config,
    stage2_config,
    split=None,
    load_target_gyro=False,
    default_dt=1.0 / 240.0,
):
    dataset_cfg = stage2_config["dataset"]
    image_cfg = stage1_config.get("image", {})
    target_cfg = stage1_config.get("target", {})
    return Stage1Stage2Dataset(
        dataset_root=dataset_cfg["root"],
        split=split or stage2_config.get("validation", {}).get("split") or dataset_cfg.get("split", "val"),
        metadata_name=dataset_cfg.get("metadata_name", "metadata.csv"),
        stage1_image_size=image_cfg.get("size", (224, 320)),
        stage1_mean=image_cfg.get("mean"),
        stage1_std=image_cfg.get("std"),
        num_vectors=target_cfg.get("num_vectors", 7),
        vector_start=target_cfg.get("vector_start", 0),
        vector_dim=target_cfg.get("vector_dim", 3),
        load_target_gyro=load_target_gyro,
        default_dt=default_dt,
        motion_field_root=dataset_cfg.get("motion_field_root"),
        motion_field_dir=dataset_cfg.get("motion_field_dir", "camera_motion_field"),
        motion_field_ext=dataset_cfg.get("motion_field_ext", "npy"),
    )


def build_stage1_stage2_loader(
    stage1_config,
    stage2_config,
    split=None,
    batch_size=None,
    num_workers=None,
    device=None,
    load_target_gyro=False,
    default_dt=1.0 / 240.0,
):
    val_cfg = stage2_config.get("validation", {})
    dataset_cfg = stage2_config.get("dataset", {})
    dataset = build_stage1_stage2_dataset(
        stage1_config,
        stage2_config,
        split=split,
        load_target_gyro=load_target_gyro,
        default_dt=default_dt,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size or val_cfg.get("batch_size", dataset_cfg.get("batch_size", 1))),
        shuffle=False,
        num_workers=int(num_workers if num_workers is not None else val_cfg.get("num_workers", 0)),
        pin_memory=device is not None and device.type == "cuda",
    )
    return dataset, loader
