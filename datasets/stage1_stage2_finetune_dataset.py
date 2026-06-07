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
        return int(image_size), int(image_size)
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


def _has_path(value):
    return bool(str(value or "").strip())


def _default_timestamp_window(num_vectors, default_dt):
    return np.arange(int(num_vectors), dtype=np.float32) * float(default_dt)


class Stage1Stage2FinetuneDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        metadata_name="metadata.csv",
        patch_size=256,
        stage1_image_size=(224, 320),
        stage1_mean=None,
        stage1_std=None,
        num_vectors=7,
        vector_start=0,
        vector_dim=3,
        default_dt=1.0 / 240.0,
        motion_field_root=None,
        motion_field_dir="camera_motion_field",
        motion_field_ext="npy",
        motion_downsample=2,
        load_target_gyro=True,
        load_target_cmf=True,
        allow_missing_gt=False,
        is_train=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.split_root = self.dataset_root / self.split
        self.patch_size = patch_size if is_train else None
        self.stage1_image_size = stage1_image_size
        self.stage1_mean = stage1_mean
        self.stage1_std = stage1_std
        self.num_vectors = int(num_vectors)
        self.vector_start = int(vector_start)
        self.vector_dim = int(vector_dim)
        self.default_dt = float(default_dt)
        self.motion_field_root = Path(motion_field_root) if motion_field_root else self.dataset_root
        self.motion_field_dir = motion_field_dir
        self.motion_field_ext = motion_field_ext
        self.motion_downsample = int(motion_downsample)
        self.load_target_gyro = bool(load_target_gyro)
        self.load_target_cmf = bool(load_target_cmf)
        self.allow_missing_gt = bool(allow_missing_gt)
        self.sensor_cache = {}
        self.timestamp_cache = {}

        patch_hw = _as_hw(self.patch_size)
        if patch_hw is not None:
            patch_h, patch_w = patch_hw
            if patch_h % self.motion_downsample != 0 or patch_w % self.motion_downsample != 0:
                raise ValueError(
                    f"patch_size must be divisible by motion_downsample: "
                    f"{patch_hw} vs {self.motion_downsample}"
                )

        self.metadata_path = find_metadata(
            self.split_root,
            metadata_name,
            ["paired_metadata.csv", "metadata.csv"],
        )
        self.fieldnames, self.rows = read_csv(self.metadata_path)
        self.layout = self._detect_layout()
        self.has_gt = self._all_rows_have_gt()
        if not self.has_gt and not self.allow_missing_gt:
            raise ValueError(
                f"{self.metadata_path} has missing sharp_path/target_sharp_path values. "
                "Set allow_missing_gt=True for inference-only real-blur data."
            )

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

    def _all_rows_have_gt(self):
        if not self.rows:
            return True
        if self.layout == "paired":
            return all(_has_path(row.get("sharp_path")) for row in self.rows)
        return all(_has_path(row.get("target_sharp_path")) for row in self.rows)

    def _image_paths(self, row):
        root = scene_root(self.split_root, row)
        if self.layout == "paired":
            gt_path = root / row["sharp_path"] if self.has_gt else None
            return root / row["blur_path"], gt_path
        gt_path = root / row["target_sharp_path"] if self.has_gt else None
        return root / row["center_blur_path"], gt_path

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

    def _load_target_cmf(self, row):
        path = self._motion_field_path(row)
        if not path.exists():
            raise FileNotFoundError(f"Missing target CMF: {path}")
        if path.suffix == ".npz":
            with np.load(path) as data:
                motion_field = data["motion_field"].astype(np.float32)
        else:
            motion_field = np.load(path).astype(np.float32)
        if motion_field.ndim != 3:
            raise ValueError(f"motion field must be HWC or CHW, got {motion_field.shape}: {path}")
        if motion_field.shape[0] <= 64 and motion_field.shape[0] < motion_field.shape[-1]:
            return torch.from_numpy(np.array(motion_field, dtype=np.float32, copy=True))
        return torch.from_numpy(np.array(motion_field.transpose(2, 0, 1), dtype=np.float32, copy=True))

    def _sample_meta(self, index, row, lq_path, gt_path):
        return {
            "index": int(index),
            "type": row.get("type", "unknown"),
            "scene_dir": row.get("scene_dir", ""),
            "stem": Path(lq_path).stem,
            "lq_path": str(lq_path),
            "gt_path": "" if gt_path is None else str(gt_path),
            "motion_field_path": str(self._motion_field_path(row)),
            "sensor_idx": self._sensor_idx(row),
        }

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lq_path, gt_path = self._image_paths(row)
        blur_full = load_image(lq_path)
        sharp_full = load_image(gt_path) if gt_path is not None else None
        stage1_image = _normalize_image(
            _resize_image(blur_full, self.stage1_image_size),
            self.stage1_mean,
            self.stage1_std,
        )
        target_cmf = self._load_target_cmf(row) if self.load_target_cmf else None

        _, image_h, image_w = blur_full.shape
        top = 0
        left = 0
        blur = blur_full
        sharp = sharp_full
        if self.patch_size is not None:
            patch_h, patch_w = _as_hw(self.patch_size)
            if patch_h > image_h or patch_w > image_w:
                raise ValueError(
                    f"Patch size {(patch_h, patch_w)} is larger than image size {(image_h, image_w)}"
                )
            max_top = (image_h - patch_h) // self.motion_downsample
            max_left = (image_w - patch_w) // self.motion_downsample
            top = int(torch.randint(0, max_top + 1, (1,)).item()) * self.motion_downsample
            left = int(torch.randint(0, max_left + 1, (1,)).item()) * self.motion_downsample
            blur = blur_full[:, top : top + patch_h, left : left + patch_w]
            if sharp is not None:
                sharp = sharp[:, top : top + patch_h, left : left + patch_w]
            if target_cmf is not None:
                mf_top = top // self.motion_downsample
                mf_left = left // self.motion_downsample
                mf_h = patch_h // self.motion_downsample
                mf_w = patch_w // self.motion_downsample
                target_cmf = target_cmf[:, mf_top : mf_top + mf_h, mf_left : mf_left + mf_w]

        sample = {
            "stage1_image": stage1_image,
            "lq": blur,
            "timestamp_window": self._load_timestamp_window(row),
            "crop_origin_yx": torch.tensor([top, left], dtype=torch.float32),
            "full_hw": torch.tensor([image_h, image_w], dtype=torch.int64),
            "meta": self._sample_meta(index, row, lq_path, gt_path),
        }
        if self.load_target_gyro:
            sample["gyro"] = self._load_target_gyro(row)
        if sharp is not None:
            sample["gt"] = sharp
        if target_cmf is not None:
            sample["target_cmf"] = target_cmf
        return sample


def build_stage1_stage2_finetune_dataset(config, stage1_config, split=None, is_train=True):
    dataset_cfg = config["dataset"]
    image_cfg = stage1_config.get("image", {})
    target_cfg = stage1_config.get("target", {})
    return Stage1Stage2FinetuneDataset(
        dataset_root=dataset_cfg["root"],
        split=split or dataset_cfg.get("split", "train"),
        metadata_name=dataset_cfg.get("metadata_name", "metadata.csv"),
        patch_size=dataset_cfg.get("patch_size", 256),
        stage1_image_size=image_cfg.get("size", (224, 320)),
        stage1_mean=image_cfg.get("mean"),
        stage1_std=image_cfg.get("std"),
        num_vectors=target_cfg.get("num_vectors", 7),
        vector_start=target_cfg.get("vector_start", 0),
        vector_dim=target_cfg.get("vector_dim", 3),
        default_dt=config.get("time", {}).get("default_dt", 1.0 / 240.0),
        motion_field_root=dataset_cfg.get("motion_field_root"),
        motion_field_dir=dataset_cfg.get("motion_field_dir", "camera_motion_field"),
        motion_field_ext=dataset_cfg.get("motion_field_ext", "npy"),
        motion_downsample=dataset_cfg.get("motion_downsample", 2),
        load_target_gyro=dataset_cfg.get("load_target_gyro", True),
        load_target_cmf=dataset_cfg.get("load_target_cmf", True),
        allow_missing_gt=dataset_cfg.get("allow_missing_gt", False),
        is_train=is_train,
    )


def build_stage1_stage2_finetune_loader(
    config,
    stage1_config,
    split=None,
    distributed=False,
    device=None,
    is_train=True,
):
    dataset_cfg = config["dataset"]
    val_cfg = config.get("validation", {})
    loader_cfg = dataset_cfg if is_train else {**dataset_cfg, **val_cfg}
    dataset_config = config
    if not is_train:
        dataset_config = {**config, "dataset": loader_cfg}
    dataset = build_stage1_stage2_finetune_dataset(
        dataset_config,
        stage1_config,
        split=split or loader_cfg.get("split"),
        is_train=is_train,
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_train)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", dataset_cfg.get("batch_size", 1))),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=int(loader_cfg.get("num_workers", dataset_cfg.get("num_workers", 0))),
        pin_memory=device is not None and device.type == "cuda",
        drop_last=is_train and bool(dataset_cfg.get("drop_last", False)),
    )
    return dataset, loader, sampler
