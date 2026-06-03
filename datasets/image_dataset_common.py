import csv

import cv2
import numpy as np
import torch


SENSOR_COLUMNS = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "grav_x",
    "grav_y",
    "grav_z",
    "cori_w",
    "cori_x",
    "cori_y",
    "cori_z",
]


def make_stratified_subset_indices(rows, limit):
    if not limit or limit >= len(rows):
        return list(range(len(rows)))

    groups = {}
    for index, row in enumerate(rows):
        key = row.get("type", "")
        groups.setdefault(key, []).append(index)

    quotas = {}
    remainders = []
    for key, indices in groups.items():
        exact = limit * len(indices) / max(len(rows), 1)
        quota = int(exact)
        quotas[key] = quota
        remainders.append((exact - quota, key))

    assigned = sum(quotas.values())
    for _remainder, key in sorted(remainders, reverse=True):
        if assigned >= limit:
            break
        quotas[key] += 1
        assigned += 1

    selected = []
    for key in sorted(groups):
        indices = groups[key]
        quota = min(quotas.get(key, 0), len(indices))
        if quota <= 0:
            continue
        if quota == len(indices):
            selected.extend(indices)
            continue
        if quota == 1:
            selected.append(indices[len(indices) // 2])
            continue

        last = len(indices) - 1
        for offset in range(quota):
            pos = round(offset * last / (quota - 1))
            selected.append(indices[pos])

    return sorted(selected[:limit])


def resolve_split_name(dataset_root, split):
    candidates = [split]
    if split == "val":
        candidates.append("validation")
    elif split == "validation":
        candidates.append("val")

    for candidate in candidates:
        if (dataset_root / candidate).exists():
            return candidate
    return candidates[0]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata csv: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def find_metadata(split_root, preferred_name, candidates):
    if preferred_name:
        path = split_root / preferred_name
        if path.exists():
            return path
        raise FileNotFoundError(f"Missing metadata csv: {path}")

    for name in candidates:
        path = split_root / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Missing metadata csv under {split_root}: {', '.join(candidates)}"
    )


def load_image(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")

    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image)


def patch_hw(patch_size):
    if patch_size is None:
        return None
    if isinstance(patch_size, int):
        return patch_size, patch_size
    return int(patch_size[0]), int(patch_size[1])


def random_crop_tensors(tensors, patch_size):
    patch_shape = patch_hw(patch_size)
    if patch_shape is None:
        return tensors

    patch_h, patch_w = patch_shape
    _, image_h, image_w = tensors[0].shape
    for tensor in tensors[1:]:
        _, tensor_h, tensor_w = tensor.shape
        if (tensor_h, tensor_w) != (image_h, image_w):
            raise ValueError(
                f"Image shapes differ: {(image_h, image_w)} vs {(tensor_h, tensor_w)}"
            )

    if patch_h > image_h or patch_w > image_w:
        raise ValueError(
            f"Patch size {(patch_h, patch_w)} is larger than image size {(image_h, image_w)}"
        )

    top = int(torch.randint(0, image_h - patch_h + 1, (1,)).item())
    left = int(torch.randint(0, image_w - patch_w + 1, (1,)).item())
    return [
        tensor[:, top : top + patch_h, left : left + patch_w]
        for tensor in tensors
    ]


def sensor_parts(sensor):
    return {
        "gyro": sensor[:, 0:3],
        "accel": sensor[:, 3:6],
        "grav": sensor[:, 6:9],
        "cori": sensor[:, 9:13],
    }


def scene_root(split_root, row):
    scene_dir = row.get("scene_dir", "")
    if scene_dir:
        return split_root / scene_dir
    return split_root


def load_scene_sensor(split_root, sensor_cache, scene_dir):
    if not scene_dir:
        raise FileNotFoundError("Sensor loading needs a scene_dir column.")

    if scene_dir not in sensor_cache:
        sensor_path = split_root / scene_dir / "sensor_windows.npy"
        if not sensor_path.exists():
            raise FileNotFoundError(f"Missing sensor_windows.npy: {sensor_path}")
        sensor_cache[scene_dir] = np.load(sensor_path, mmap_mode="r")
    return sensor_cache[scene_dir]
