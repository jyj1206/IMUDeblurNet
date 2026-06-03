from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.image_dataset_common import (
    SENSOR_COLUMNS,
    find_metadata,
    load_image,
    load_scene_sensor,
    random_crop_tensors,
    read_csv,
    resolve_split_name,
    scene_root,
    sensor_parts,
)


class TripletImageDataset(Dataset):
    def __init__(self, dataset_root, split, use_imu_value=False, patch_size=256, metadata_name=None):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.use_imu_value = bool(use_imu_value)
        self.patch_size = patch_size if self.split == "train" else None

        self.split_root = self.dataset_root / self.split
        self.metadata_path = find_metadata(
            self.split_root,
            metadata_name,
            ["triplet_metadata.csv", "metadata.csv"],
        )
        self.fieldnames, self.rows = read_csv(self.metadata_path)
        self._validate_triplet_metadata()
        self.sensor_cache = {}

    def _validate_triplet_metadata(self):
        if not self.rows:
            return

        required = {
            "scene_dir",
            "prev_blur_path",
            "center_blur_path",
            "next_blur_path",
            "target_sharp_path",
        }
        columns = set(self.rows[0].keys())
        if not required.issubset(columns):
            raise ValueError(
                f"{self.metadata_path} must contain columns: {', '.join(sorted(required))}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        root = scene_root(self.split_root, row)

        prev_lq = load_image(root / row["prev_blur_path"])
        center_lq = load_image(root / row["center_blur_path"])
        next_lq = load_image(root / row["next_blur_path"])
        gt = load_image(root / row["target_sharp_path"])

        prev_lq, center_lq, next_lq, gt = random_crop_tensors(
            [prev_lq, center_lq, next_lq, gt],
            self.patch_size,
        )

        sample = {
            "lq": center_lq,
            "gt": gt,
            "prev_lq": prev_lq,
            "center_lq": center_lq,
            "next_lq": next_lq,
            "lq_triplet": torch.stack([prev_lq, center_lq, next_lq], dim=0),
        }

        if self.use_imu_value:
            sensor_windows = load_scene_sensor(
                self.split_root,
                self.sensor_cache,
                row["scene_dir"],
            )
            sensor_indices = [
                int(row["prev_sensor_idx"]),
                int(row["center_sensor_idx"]),
                int(row["next_sensor_idx"]),
            ]
            imu_triplet = torch.from_numpy(
                np.array(sensor_windows[sensor_indices], dtype=np.float32, copy=True)
            )
            imu = imu_triplet[1]
            parts = sensor_parts(imu)
            flat_parts = sensor_parts(imu_triplet.reshape(-1, imu_triplet.shape[-1]))
            sample.update(
                {
                    "imu": imu,
                    "imu_triplet": imu_triplet,
                    "gyro": parts["gyro"],
                    "accel": parts["accel"],
                    "grav": parts["grav"],
                    "cori": parts["cori"],
                    "gyro_triplet": flat_parts["gyro"].reshape(3, imu_triplet.shape[1], 3),
                    "accel_triplet": flat_parts["accel"].reshape(3, imu_triplet.shape[1], 3),
                    "grav_triplet": flat_parts["grav"].reshape(3, imu_triplet.shape[1], 3),
                    "cori_triplet": flat_parts["cori"].reshape(3, imu_triplet.shape[1], 4),
                }
            )

        return sample


Dataset_TripletImage = TripletImageDataset
