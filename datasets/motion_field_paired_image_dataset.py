from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .image_dataset_common import (
    find_metadata,
    load_image,
    read_csv,
    resolve_split_name,
    scene_root,
)


class MotionFieldPairedImageDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split,
        patch_size=256,
        metadata_name=None,
        motion_field_root=None,
        motion_field_dir="camera_motion_field",
        motion_field_ext="npy",
        motion_downsample=2,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.patch_size = patch_size if self.split == "train" else None
        self.motion_field_root = Path(motion_field_root) if motion_field_root else self.dataset_root
        self.motion_field_dir = motion_field_dir
        self.motion_field_ext = motion_field_ext
        self.motion_downsample = int(motion_downsample)
        if self.patch_size and self.patch_size % self.motion_downsample != 0:
            raise ValueError(
                f"patch_size must be divisible by motion_downsample: "
                f"{self.patch_size} vs {self.motion_downsample}"
            )

        self.split_root = self.dataset_root / self.split
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
            f"{self.metadata_path} must contain blur_path/sharp_path for paired data."
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

    def _sample_meta(self, index, row, lq_path, gt_path):
        motion_path = self._motion_field_path(row)
        return {
            "index": int(index),
            "type": row.get("type", "unknown"),
            "scene_dir": row.get("scene_dir", ""),
            "stem": Path(lq_path).stem,
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
            "motion_field_path": str(motion_path),
        }

    def _load_motion_field(self, row):
        path = self._motion_field_path(row)
        if not path.exists():
            raise FileNotFoundError(f"Missing motion field: {path}")

        if path.suffix == ".npz":
            with np.load(path) as data:
                motion_field = data["motion_field"].astype(np.float32)
        else:
            motion_field = np.load(path).astype(np.float32)
        if motion_field.ndim != 3:
            raise ValueError(f"motion field must be HWC or CHW, got {motion_field.shape}")
        if motion_field.shape[0] <= 32 and motion_field.shape[0] < motion_field.shape[-1]:
            return torch.from_numpy(np.ascontiguousarray(motion_field))
        return torch.from_numpy(np.ascontiguousarray(motion_field.transpose(2, 0, 1)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lq_path, gt_path = self._image_paths(row)
        lq = load_image(lq_path)
        gt = load_image(gt_path)
        motion_field = self._load_motion_field(row)

        if self.patch_size:
            _, image_h, image_w = lq.shape
            if self.patch_size > image_h or self.patch_size > image_w:
                raise ValueError(
                    f"Patch size {self.patch_size} is larger than image size {(image_h, image_w)}"
                )
            top = int(torch.randint(0, image_h - self.patch_size + 1, (1,)).item())
            left = int(torch.randint(0, image_w - self.patch_size + 1, (1,)).item())
            lq = lq[:, top : top + self.patch_size, left : left + self.patch_size]
            gt = gt[:, top : top + self.patch_size, left : left + self.patch_size]

            mf_top = top // self.motion_downsample
            mf_left = left // self.motion_downsample
            mf_size = self.patch_size // self.motion_downsample
            motion_field = motion_field[
                :, mf_top : mf_top + mf_size, mf_left : mf_left + mf_size
            ]

        return {
            "lq": lq,
            "gt": gt,
            "motion_field": motion_field,
            "meta": self._sample_meta(index, row, lq_path, gt_path),
        }


Dataset_MotionFieldPairedImage = MotionFieldPairedImageDataset
