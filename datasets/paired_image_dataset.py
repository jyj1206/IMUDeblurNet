from pathlib import Path

from torch.utils.data import Dataset

from .image_dataset_common import (
    find_metadata,
    load_image,
    random_crop_tensors,
    read_csv,
    resolve_split_name,
    scene_root,
)


class PairedImageDataset(Dataset):
    def __init__(self, dataset_root, split, patch_size=256, metadata_name=None):
        self.dataset_root = Path(dataset_root)
        self.split = resolve_split_name(self.dataset_root, split)
        self.patch_size = patch_size if self.split == "train" else None

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

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        lq_path, gt_path = self._image_paths(row)
        lq, gt = random_crop_tensors(
            [load_image(lq_path), load_image(gt_path)],
            self.patch_size,
        )

        return {
            "lq": lq,
            "gt": gt,
        }


Dataset_PairedImage = PairedImageDataset
