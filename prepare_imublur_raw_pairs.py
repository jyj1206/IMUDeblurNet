import argparse
import csv
import os
import shutil
from pathlib import Path

from tqdm import tqdm


SPLITS = ("train", "val", "test")


def copy_file(source, target, method):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return False
    if method == "copy":
        shutil.copy2(source, target)
    elif method == "hardlink":
        os.link(source, target)
    elif method == "symlink":
        os.symlink(source, target)
    else:
        raise ValueError(f"Unknown method: {method}")
    return True


def read_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def source_paths(split_root, row):
    scene_root = split_root / row["scene_dir"]
    if row.get("blur_path") and row.get("sharp_path"):
        return scene_root / row["blur_path"], scene_root / row["sharp_path"]
    if row.get("center_blur_path") and row.get("target_sharp_path"):
        return scene_root / row["center_blur_path"], scene_root / row["target_sharp_path"]
    raise ValueError("metadata row needs blur_path/sharp_path or center_blur_path/target_sharp_path")


def safe_stem(split, row, source_blur, include_split=True):
    scene = row["scene_dir"]
    stem = Path(row.get("sample_name") or source_blur.name).stem
    if include_split:
        return f"{split}_{scene}_{stem}"
    return f"{scene}_{stem}"


def target_paths(output_root, split, row, source_blur, source_sharp, layout):
    motion_type = str(row.get("type") or "unknown").strip() or "unknown"
    if layout == "flat":
        stem = safe_stem(split, row, source_blur, include_split=True)
        return (
            output_root / "blur" / f"{stem}{source_blur.suffix.lower()}",
            output_root / "sharp" / f"{stem}{source_sharp.suffix.lower()}",
            motion_type,
        )
    if layout == "grouped":
        stem = safe_stem(split, row, source_blur, include_split=False)
        return (
            output_root / split / motion_type / "blur" / f"{stem}{source_blur.suffix.lower()}",
            output_root / split / motion_type / "sharp" / f"{stem}{source_sharp.suffix.lower()}",
            motion_type,
        )
    raise ValueError(f"Unknown layout: {layout}")


def prepare_raw_pairs(args):
    data_root = args.data_root
    output_root = args.output_root or data_root / "raw"
    metadata_path = output_root / args.metadata_name

    output_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    missing = []
    copied = 0
    skipped = 0

    for split in args.splits:
        split_root = data_root / split
        metadata_file = split_root / args.metadata_name
        if not metadata_file.exists():
            if args.strict:
                raise FileNotFoundError(f"Missing metadata: {metadata_file}")
            print(f"skip missing split metadata: {metadata_file}")
            continue

        rows = read_rows(metadata_file)
        if args.max_samples:
            rows = rows[: args.max_samples]

        for row in tqdm(rows, desc=f"{split} raw pairs"):
            source_blur, source_sharp = source_paths(split_root, row)
            if not source_blur.exists() or not source_sharp.exists():
                missing.append((split, row.get("scene_dir", ""), str(source_blur), str(source_sharp)))
                if args.strict:
                    raise FileNotFoundError(f"Missing pair: {source_blur}, {source_sharp}")
                continue

            blur_target, sharp_target, motion_type = target_paths(
                output_root,
                split,
                row,
                source_blur,
                source_sharp,
                args.layout,
            )

            blur_created = copy_file(source_blur, blur_target, args.method)
            sharp_created = copy_file(source_sharp, sharp_target, args.method)
            copied += int(blur_created) + int(sharp_created)
            skipped += int(not blur_created) + int(not sharp_created)

            output_rows.append(
                {
                    "split": split,
                    "type": motion_type,
                    "scene_dir": row["scene_dir"],
                    "sample_name": row.get("sample_name") or source_blur.name,
                    "blur_path": str(blur_target.relative_to(output_root)).replace("\\", "/"),
                    "sharp_path": str(sharp_target.relative_to(output_root)).replace("\\", "/"),
                    "source_blur_path": str(source_blur.relative_to(data_root)).replace("\\", "/"),
                    "source_sharp_path": str(source_sharp.relative_to(data_root)).replace("\\", "/"),
                }
            )

    with metadata_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "split",
            "type",
            "scene_dir",
            "sample_name",
            "blur_path",
            "sharp_path",
            "source_blur_path",
            "source_sharp_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"output_root: {output_root}")
    print(f"pairs: {len(output_rows)}")
    print(f"created file entries: {copied}")
    print(f"existing/skipped checks: {skipped}")
    print(f"missing pairs: {len(missing)}")
    print(f"metadata: {metadata_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata-name", default="metadata.csv")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--layout", choices=["grouped", "flat"], default="grouped")
    parser.add_argument("--method", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main():
    prepare_raw_pairs(parse_args())


if __name__ == "__main__":
    main()
