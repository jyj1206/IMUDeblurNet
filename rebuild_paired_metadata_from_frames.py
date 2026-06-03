import argparse
import csv
from pathlib import Path


PREFERRED_COLUMNS = [
    "sample_idx",
    "scene_dir",
    "local_frame_idx",
    "sample_name",
    "split",
    "type",
    "type_id",
    "video_id",
    "clip_part",
    "target_frame_number",
    "blur_path",
    "sharp_path",
    "sensor_idx",
    "timestamp_idx",
    "original_global_idx",
    "window_size",
    "gamma",
    "source_start_frame",
    "source_end_frame",
    "target_frame",
    "source_frames",
    "sensor_frame_indices",
    "original_blur_path",
    "original_sharp_path",
]


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or [], list(reader)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_fieldnames(rows):
    fields = []
    seen = set()
    for column in PREFERRED_COLUMNS:
        if any(column in row for row in rows):
            fields.append(column)
            seen.add(column)
    for row in rows:
        for column in row:
            if column not in seen:
                fields.append(column)
                seen.add(column)
    return fields


def scene_dirs(split_root):
    return sorted(
        path
        for path in split_root.glob("scene_*")
        if path.is_dir() and (path / "frames.csv").exists()
    )


def build_split_metadata(split_root):
    rows = []
    missing_required = []
    for scene_dir in scene_dirs(split_root):
        fieldnames, frame_rows = read_csv(scene_dir / "frames.csv")
        required = ["blur_path", "sharp_path", "sensor_idx"]
        missing = [column for column in required if column not in fieldnames]
        if missing:
            missing_required.append((scene_dir, missing))
            continue

        for frame_row in frame_rows:
            row = {"sample_idx": str(len(rows)), "scene_dir": scene_dir.name}
            row.update(frame_row)
            rows.append(row)

    if missing_required:
        details = "; ".join(f"{path.name}: {missing}" for path, missing in missing_required[:5])
        raise ValueError(f"Missing required frame columns: {details}")

    return ordered_fieldnames(rows), rows


def remove_triplet_files(split_root, apply):
    triplet_paths = sorted(split_root.glob("scene_*/triplets.csv"))
    if apply:
        for path in triplet_paths:
            path.unlink()
    return triplet_paths


def rebuild_split(dataset_root, split, metadata_name, apply, remove_triplets):
    split_root = dataset_root / split
    if not split_root.exists():
        return {
            "split": split,
            "exists": False,
            "rows": 0,
            "scenes": 0,
            "metadata_path": str(split_root / metadata_name),
            "removed_triplets": 0,
        }

    fieldnames, rows = build_split_metadata(split_root)
    metadata_path = split_root / metadata_name
    triplet_paths = remove_triplet_files(split_root, apply and remove_triplets)

    if apply:
        write_csv(metadata_path, fieldnames, rows)

    return {
        "split": split,
        "exists": True,
        "rows": len(rows),
        "scenes": len(scene_dirs(split_root)),
        "metadata_path": str(metadata_path),
        "removed_triplets": len(triplet_paths) if remove_triplets else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild split metadata.csv from each scene's frames.csv so paired "
            "datasets use every sample instead of center-only triplets."
        )
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/IMUBlur"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--metadata-name", default="metadata.csv")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--remove-triplets",
        action="store_true",
        help="Delete scene_*/triplets.csv after rebuilding metadata. Only active with --apply.",
    )
    args = parser.parse_args()

    results = [
        rebuild_split(
            dataset_root=args.dataset_root,
            split=split,
            metadata_name=args.metadata_name,
            apply=args.apply,
            remove_triplets=args.remove_triplets,
        )
        for split in args.splits
    ]

    action = "wrote" if args.apply else "would_write"
    for result in results:
        if not result["exists"]:
            print(f"{result['split']}: missing split")
            continue
        print(
            f"{result['split']}: {action} {result['rows']} rows "
            f"from {result['scenes']} scenes -> {result['metadata_path']}"
        )
        if args.remove_triplets:
            removed_action = "removed" if args.apply else "would_remove"
            print(f"  {removed_action} triplets.csv files: {result['removed_triplets']}")

    if not args.apply:
        print("dry-run only. Add --apply to rewrite metadata.csv.")


if __name__ == "__main__":
    raise SystemExit(main())
