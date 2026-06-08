import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def os_relpath(path, start):
    import os

    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def image_files(directory):
    return sorted(
        path
        for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def ensure_empty_or_overwrite(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_pair(src_blur, src_sharp, dst_scene, blur_path, sharp_path):
    dst_blur = dst_scene / blur_path
    dst_sharp = dst_scene / sharp_path
    dst_blur.parent.mkdir(parents=True, exist_ok=True)
    dst_sharp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_blur, dst_blur)
    shutil.copy2(src_sharp, dst_sharp)


def row_for_pair(
    split_root, source_scene, src_blur, src_sharp, dataset_type, storage, scene_name
):
    if storage == "metadata":
        scene_dir = os_relpath(source_scene, split_root)
        blur_path = os_relpath(src_blur, source_scene)
        sharp_path = os_relpath(src_sharp, source_scene)
    else:
        scene_dir = scene_name
        blur_path = f"blur/{src_blur.name}"
        sharp_path = f"sharp/{src_sharp.name}"

    return {
        "scene_dir": scene_dir,
        "blur_path": blur_path,
        "sharp_path": sharp_path,
        "type": dataset_type,
    }


def collect_gopro_rows(source_root, output_root, split, storage):
    split_root = output_root / split
    rows = []
    for scene in sorted(path for path in source_root.iterdir() if path.is_dir()):
        blur_dir = scene / "blur"
        sharp_dir = scene / "sharp"
        if not blur_dir.exists() or not sharp_dir.exists():
            continue

        sharp_by_name = {path.name: path for path in image_files(sharp_dir)}
        for blur_path in image_files(blur_dir):
            sharp_path = sharp_by_name.get(blur_path.name)
            if sharp_path is None:
                continue
            row = row_for_pair(
                split_root=split_root,
                source_scene=scene,
                src_blur=blur_path,
                src_sharp=sharp_path,
                dataset_type="gopro",
                storage=storage,
                scene_name=scene.name,
            )
            rows.append((row, scene, blur_path, sharp_path))
    return rows


def parse_realblur_list_line(line):
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    sharp_rel, blur_rel = parts[0], parts[1]
    return sharp_rel, blur_rel


def collect_realblur_rows(source_root, output_root, split, storage, list_file):
    split_root = output_root / split
    rows = []

    if list_file:
        with Path(list_file).open("r", encoding="utf-8-sig") as file:
            pairs = [parse_realblur_list_line(line) for line in file]
        pairs = [pair for pair in pairs if pair is not None]
        for sharp_rel, blur_rel in pairs:
            src_sharp = source_root / sharp_rel
            src_blur = source_root / blur_rel
            if not src_blur.exists() or not src_sharp.exists():
                raise FileNotFoundError(
                    f"Missing RealBlur pair: {src_blur}, {src_sharp}"
                )
            source_scene = src_blur.parents[1]
            scene_name = source_scene.name
            row = row_for_pair(
                split_root=split_root,
                source_scene=source_scene,
                src_blur=src_blur,
                src_sharp=src_sharp,
                dataset_type="realblur_j",
                storage=storage,
                scene_name=scene_name,
            )
            rows.append((row, source_scene, src_blur, src_sharp))
        return rows

    for scene in sorted(path for path in source_root.iterdir() if path.is_dir()):
        blur_dir = scene / "blur"
        sharp_dir = scene / "gt"
        if not blur_dir.exists() or not sharp_dir.exists():
            continue
        sharp_by_index = {
            path.stem.removeprefix("gt_"): path for path in image_files(sharp_dir)
        }
        for blur_path in image_files(blur_dir):
            key = blur_path.stem.removeprefix("blur_")
            sharp_path = sharp_by_index.get(key)
            if sharp_path is None:
                continue
            row = row_for_pair(
                split_root=split_root,
                source_scene=scene,
                src_blur=blur_path,
                src_sharp=sharp_path,
                dataset_type="realblur_j",
                storage=storage,
                scene_name=scene.name,
            )
            rows.append((row, scene, blur_path, sharp_path))
    return rows


def write_metadata(metadata_path, rows, overwrite):
    ensure_empty_or_overwrite(metadata_path, overwrite)
    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scene_dir", "blur_path", "sharp_path", "type"],
        )
        writer.writeheader()
        for row, _scene, _blur, _sharp in rows:
            writer.writerow(row)


def materialize_copy(output_root, split, rows):
    split_root = output_root / split
    for row, _scene, blur_path, sharp_path in rows:
        dst_scene = split_root / row["scene_dir"]
        copy_pair(
            blur_path,
            sharp_path,
            dst_scene,
            Path(row["blur_path"]),
            Path(row["sharp_path"]),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert paired deblurring datasets to this project's metadata format."
    )
    parser.add_argument("--dataset", choices=["gopro", "realblur_j"], required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--metadata-name", default="metadata.csv")
    parser.add_argument(
        "--storage",
        choices=["metadata", "copy"],
        default="metadata",
        help="metadata stores only a CSV that points to original files; copy duplicates images.",
    )
    parser.add_argument(
        "--list-file",
        type=Path,
        help="Optional RealBlur list file. Lines are expected as: gt_path blur_path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_root
    split_root = output_root / args.split
    metadata_path = split_root / args.metadata_name

    if args.dataset == "gopro":
        rows = collect_gopro_rows(
            source_root=args.source_root,
            output_root=output_root,
            split=args.split,
            storage=args.storage,
        )
    else:
        rows = collect_realblur_rows(
            source_root=args.source_root,
            output_root=output_root,
            split=args.split,
            storage=args.storage,
            list_file=args.list_file,
        )

    if not rows:
        raise RuntimeError(f"No pairs found under {args.source_root}")

    if args.storage == "copy":
        materialize_copy(output_root, args.split, rows)

    write_metadata(metadata_path, rows, args.overwrite)
    print(f"saved {len(rows)} rows: {metadata_path}")
    print(f"storage={args.storage}")


if __name__ == "__main__":
    main()
