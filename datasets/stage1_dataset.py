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


class Stage1Dataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        metadata_name="metadata.csv",
        image_size=(224, 320),
        normalize_mean=None,
        normalize_std=None,
        num_vectors=7,
        vector_start=0,
        vector_dim=3,
        default_dt=1.0 / 240.0,
        camera=None,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.split_root = self.dataset_root / self.split
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_vectors = int(num_vectors)
        self.vector_start = int(vector_start)
        self.vector_dim = int(vector_dim)
        self.sensor_cache = {}

        self.metadata_path = find_metadata(
            self.split_root,
            metadata_name,
            ["paired_metadata.csv", "metadata.csv"],
        )
        self.fieldnames, self.rows = read_csv(self.metadata_path)
        self.layout = self._detect_layout()

        camera = camera or {}
        self.default_dt = float(default_dt)
        self.native_size = tuple(camera.get("native_size", [1080, 1920]))
        self.fx = float(camera.get("fx", 923.7181693))
        self.fy = float(camera.get("fy", 924.51235192))
        self.focal_length = camera.get("focal_length")
        self.timestamp_cache = {}

    def _detect_layout(self):
        if not self.rows:
            return "paired"
        columns = set(self.rows[0].keys())
        if "blur_path" in columns:
            return "paired"
        if "center_blur_path" in columns:
            return "triplet_center_fallback"
        raise ValueError(
            f"{self.metadata_path} must contain blur_path or center_blur_path."
        )

    def _blur_path(self, row):
        root = scene_root(self.split_root, row)
        if self.layout == "paired":
            return root / row["blur_path"]
        return root / row["center_blur_path"]

    def _sensor_idx(self, row):
        return int(row.get("sensor_idx") or row.get("center_sensor_idx") or 0)

    def _load_sensor_windows(self, row):
        scene_dir = row.get("scene_dir", "")
        if not scene_dir:
            raise FileNotFoundError("Stage1 gyro target needs scene_dir in metadata.")
        if scene_dir not in self.sensor_cache:
            path = self.split_root / scene_dir / "sensor_windows.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing sensor_windows.npy: {path}")
            self.sensor_cache[scene_dir] = np.load(path, mmap_mode="r")
        return self.sensor_cache[scene_dir]

    def _load_gyro_window(self, row):
        sensor_windows = self._load_sensor_windows(row)
        sensor_idx = self._sensor_idx(row)
        end = self.vector_start + self.vector_dim
        gyro = np.asarray(
            sensor_windows[sensor_idx, : self.num_vectors, self.vector_start : end],
            dtype=np.float32,
        )
        if gyro.shape != (self.num_vectors, self.vector_dim):
            raise ValueError(
                f"gyro target shape must be {(self.num_vectors, self.vector_dim)}, got {gyro.shape}"
            )
        return torch.from_numpy(np.array(gyro, dtype=np.float32, copy=True))

    def _load_timestamp_window(self, row):
        scene_dir = row.get("scene_dir", "")
        if not scene_dir:
            window = _default_timestamp_window(self.num_vectors, self.default_dt)
            return torch.from_numpy(window)

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
            window = np.asarray(
                timestamps[sensor_idx, : self.num_vectors], dtype=np.float32
            )
            if window.shape[0] != self.num_vectors:
                window = _default_timestamp_window(self.num_vectors, self.default_dt)
        return torch.from_numpy(np.array(window, dtype=np.float32, copy=True))

    def _scaled_focal_length(self):
        if self.focal_length is not None:
            return float(self.focal_length)
        image_size = self.image_size
        if image_size is None:
            height, width = self.native_size
        else:
            height, width = int(image_size[0]), int(image_size[1])
        native_h, native_w = float(self.native_size[0]), float(self.native_size[1])
        fx = self.fx * (float(width) / native_w)
        fy = self.fy * (float(height) / native_h)
        return 0.5 * (fx + fy)

    def _sample_meta(self, index, row, image_path):
        return {
            "index": int(index),
            "type": row.get("type", "unknown"),
            "scene_dir": row.get("scene_dir", ""),
            "stem": Path(image_path).stem,
            "image_path": str(image_path),
            "sensor_idx": self._sensor_idx(row),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self._blur_path(row)
        image = load_image(image_path)
        image = _resize_image(image, self.image_size)
        image = _normalize_image(image, self.normalize_mean, self.normalize_std)
        sample = {
            "image": image,
            "gyro": self._load_gyro_window(row),
            "timestamp_window": self._load_timestamp_window(row),
            "focal_length": torch.tensor(
                self._scaled_focal_length(), dtype=torch.float32
            ),
            "meta": self._sample_meta(index, row, image_path),
        }
        return sample


def build_stage1_dataset(config, split=None):
    dataset_cfg = config["dataset"]
    target_cfg = config.get("target", {})
    image_cfg = config.get("image", {})
    camera_cfg = config.get("camera", {})
    return Stage1Dataset(
        dataset_root=dataset_cfg["root"],
        split=split or dataset_cfg.get("split", "train"),
        metadata_name=dataset_cfg.get("metadata_name", "metadata.csv"),
        image_size=image_cfg.get("size", (224, 320)),
        normalize_mean=image_cfg.get("mean"),
        normalize_std=image_cfg.get("std"),
        num_vectors=target_cfg.get("num_vectors", 7),
        vector_start=target_cfg.get("vector_start", 0),
        vector_dim=target_cfg.get("vector_dim", 3),
        default_dt=config.get("time", {}).get("default_dt", 1.0 / 240.0),
        camera=camera_cfg,
    )


def build_stage1_loader(
    config, split=None, distributed=False, device=None, is_train=True
):
    dataset_cfg = config["dataset"]
    loader_cfg = (
        dataset_cfg if is_train else {**dataset_cfg, **config.get("validation", {})}
    )
    dataset = build_stage1_dataset(config, split=split or loader_cfg.get("split"))
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", dataset_cfg.get("batch_size", 8))),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=int(
            loader_cfg.get("num_workers", dataset_cfg.get("num_workers", 0))
        ),
        pin_memory=device is not None and device.type == "cuda",
        drop_last=is_train and bool(dataset_cfg.get("drop_last", False)),
    )
    return dataset, loader, sampler
