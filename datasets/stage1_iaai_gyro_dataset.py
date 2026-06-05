from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .stage1_gyro_dataset import Stage1GyroDataset


def _default_timestamp_window(num_vectors, default_dt):
    return np.arange(int(num_vectors), dtype=np.float32) * float(default_dt)


class Stage1IAAIGyroDataset(Stage1GyroDataset):
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
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            metadata_name=metadata_name,
            image_size=image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            num_vectors=num_vectors,
            vector_start=vector_start,
            vector_dim=vector_dim,
        )
        camera = camera or {}
        self.default_dt = float(default_dt)
        self.native_size = tuple(camera.get("native_size", [1080, 1920]))
        self.fx = float(camera.get("fx", 923.7181693))
        self.fy = float(camera.get("fy", 924.51235192))
        self.focal_length = camera.get("focal_length")
        self.timestamp_cache = {}

    def _load_timestamp_window(self, row):
        scene_dir = row.get("scene_dir", "")
        if not scene_dir:
            window = _default_timestamp_window(self.num_vectors, self.default_dt)
            return torch.from_numpy(window)

        if scene_dir not in self.timestamp_cache:
            path = self.split_root / scene_dir / "sensor_timestamps.npy"
            self.timestamp_cache[scene_dir] = np.load(path, mmap_mode="r") if path.exists() else None

        timestamps = self.timestamp_cache[scene_dir]
        if timestamps is None:
            window = _default_timestamp_window(self.num_vectors, self.default_dt)
        else:
            sensor_idx = self._sensor_idx(row)
            window = np.asarray(timestamps[sensor_idx, : self.num_vectors], dtype=np.float32)
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

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.rows[index]
        sample["timestamp_window"] = self._load_timestamp_window(row)
        sample["focal_length"] = torch.tensor(self._scaled_focal_length(), dtype=torch.float32)
        return sample


def build_stage1_iaai_dataset(config, split=None):
    dataset_cfg = config["dataset"]
    target_cfg = config.get("target", {})
    image_cfg = config.get("image", {})
    camera_cfg = config.get("camera", {})
    return Stage1IAAIGyroDataset(
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


def build_stage1_iaai_loader(config, split=None, distributed=False, device=None, is_train=True):
    dataset_cfg = config["dataset"]
    loader_cfg = dataset_cfg if is_train else {**dataset_cfg, **config.get("validation", {})}
    dataset = build_stage1_iaai_dataset(config, split=split or loader_cfg.get("split"))
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
        num_workers=int(loader_cfg.get("num_workers", dataset_cfg.get("num_workers", 0))),
        pin_memory=device is not None and device.type == "cuda",
        drop_last=is_train and bool(dataset_cfg.get("drop_last", False)),
    )
    return dataset, loader, sampler
