import torch
from torch.utils.data import DataLoader

from .motion_field_paired_image_dataset import MotionFieldPairedImageDataset


def build_dataset(config, split=None):
    data_cfg = dict(config["dataset"])
    if split is not None:
        data_cfg["split"] = split

    return MotionFieldPairedImageDataset(
        dataset_root=data_cfg["root"],
        split=data_cfg.get("split", "train"),
        metadata_name=data_cfg.get("metadata_name", "metadata.csv"),
        patch_size=data_cfg.get("patch_size", 256),
        motion_field_root=data_cfg.get("motion_field_root"),
        motion_field_dir=data_cfg.get("motion_field_dir", "camera_motion_field"),
        motion_field_ext=data_cfg.get("motion_field_ext", "npy"),
        motion_downsample=data_cfg.get("motion_downsample", 2),
    )


def build_dataloader(config, split=None, distributed=False, device=None, is_train=True):
    data_cfg = config["dataset"]
    loader_cfg = data_cfg if is_train else {**data_cfg, **config.get("validation", {})}
    dataset = build_dataset(config, split=split or loader_cfg.get("split"))
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=is_train,
    ) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", data_cfg.get("batch_size", 4))),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=int(loader_cfg.get("num_workers", data_cfg.get("num_workers", 0))),
        pin_memory=device is not None and device.type == "cuda",
        drop_last=is_train and bool(data_cfg.get("drop_last", False)),
    )
    return dataset, loader, sampler
