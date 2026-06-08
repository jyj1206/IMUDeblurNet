import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("data") / "undisroted_image"
DEFAULT_OUTPUT_DIR = Path("result") / "undistort"
DEFAULT_PARAMS_PATH = Path("result") / "calibration" / "calibration_params.npz"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def collect_images(image_dir):
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(pattern))
        paths.extend(image_dir.glob(pattern.upper()))

    filtered = []
    for path in sorted(set(paths)):
        lowered = path.stem.lower()
        if "undistort" in lowered or "comparison" in lowered:
            continue
        filtered.append(path)
    return filtered


def load_params(params_path):
    if not params_path.exists():
        raise FileNotFoundError(
            f"Calibration parameter file not found: {params_path}. Run cali.py first."
        )

    data = np.load(params_path)
    return {
        "calibration_size": (int(data["image_width"]), int(data["image_height"])),
        "rms_standard": float(data["rms_standard"]),
        "camera_matrix_standard": data["camera_matrix_standard"].astype(np.float64),
        "dist_coeffs_standard": data["dist_coeffs_standard"].astype(np.float64),
        "rms_fisheye": float(data["rms_fisheye"]),
        "camera_matrix_fisheye": data["camera_matrix_fisheye"].astype(np.float64),
        "dist_coeffs_fisheye": data["dist_coeffs_fisheye"].astype(np.float64),
    }


def scaled_camera_matrix(camera_matrix, calibration_size, image_size):
    if calibration_size == image_size:
        return camera_matrix.copy()

    scale_x = image_size[0] / calibration_size[0]
    scale_y = image_size[1] / calibration_size[1]
    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def undistort_standard(image, camera_matrix, dist_coeffs, alpha, crop):
    height, width = image.shape[:2]
    image_size = (width, height)
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )
    output = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    if crop:
        x, y, roi_width, roi_height = roi
        if roi_width > 0 and roi_height > 0:
            output = output[y : y + roi_height, x : x + roi_width]

    return output


def undistort_fisheye(image, camera_matrix, dist_coeffs, balance, fov_scale, new_camera_mode):
    height, width = image.shape[:2]
    image_size = (width, height)
    dist_coeffs = dist_coeffs.reshape(4, 1)

    if new_camera_mode == "same":
        new_camera_matrix = camera_matrix
    else:
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist_coeffs,
            image_size,
            np.eye(3),
            balance=balance,
            new_size=image_size,
            fov_scale=fov_scale,
        )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        np.eye(3),
        new_camera_matrix,
        image_size,
        cv2.CV_16SC2,
    )
    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def label_image(image, text, color):
    output = image.copy()
    cv2.putText(
        output,
        text,
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA,
    )
    return output


def save_comparison(original, standard, fisheye, output_path, visual_scale):
    height, width = original.shape[:2]
    target_size = (width, height)
    standard_view = cv2.resize(standard, target_size)
    fisheye_view = cv2.resize(fisheye, target_size)

    views = [
        label_image(original, "Original", (0, 0, 255)),
        label_image(standard_view, "Standard", (0, 180, 0)),
        label_image(fisheye_view, "Fisheye", (255, 0, 0)),
    ]
    if visual_scale != 1.0:
        scaled_size = (int(width * visual_scale), int(height * visual_scale))
        views = [cv2.resize(view, scaled_size) for view in views]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.hstack(views))


def make_output_dirs(output_dir):
    dirs = {
        "standard": output_dir / "standard",
        "fisheye": output_dir / "fisheye",
        "comparison": output_dir / "comparison",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def process_image(
    image_path,
    output_dirs,
    params,
    model,
    alpha,
    balance,
    fov_scale,
    fisheye_new_camera,
    crop,
    save_side_by_side,
    visual_scale,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipped unreadable image: {image_path}")
        return []

    height, width = image.shape[:2]
    image_size = (width, height)
    calibration_size = params["calibration_size"]
    output_paths = []
    standard_output = None
    fisheye_output = None

    if model in ("standard", "both"):
        matrix = scaled_camera_matrix(
            params["camera_matrix_standard"],
            calibration_size,
            image_size,
        )
        standard_output = undistort_standard(
            image,
            matrix,
            params["dist_coeffs_standard"],
            alpha,
            crop,
        )
        output_path = output_dirs["standard"] / f"{image_path.stem}_standard{image_path.suffix}"
        cv2.imwrite(str(output_path), standard_output)
        output_paths.append(output_path)

    if model in ("fisheye", "both"):
        matrix = scaled_camera_matrix(
            params["camera_matrix_fisheye"],
            calibration_size,
            image_size,
        )
        fisheye_output = undistort_fisheye(
            image,
            matrix,
            params["dist_coeffs_fisheye"],
            balance,
            fov_scale,
            fisheye_new_camera,
        )
        output_path = output_dirs["fisheye"] / f"{image_path.stem}_fisheye{image_path.suffix}"
        cv2.imwrite(str(output_path), fisheye_output)
        output_paths.append(output_path)

    if save_side_by_side and standard_output is not None and fisheye_output is not None:
        comparison_path = output_dirs["comparison"] / f"{image_path.stem}_comparison.jpg"
        save_comparison(image, standard_output, fisheye_output, comparison_path, visual_scale)
        output_paths.append(comparison_path)

    return output_paths


def build_parser():
    parser = argparse.ArgumentParser(
        description="Undistort images with calibration parameters from cali.py.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument(
        "--model",
        choices=("standard", "fisheye", "both"),
        default="both",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="Standard model free scaling, 0 reduces black borders and 1 keeps all pixels.")
    parser.add_argument("--balance", type=float, default=0.0, help="Fisheye balance, 0 crops black borders and 1 keeps wider FOV.")
    parser.add_argument("--fov-scale", type=float, default=1.0)
    parser.add_argument(
        "--fisheye-new-camera",
        choices=("same", "estimate"),
        default="same",
        help="Use original K for fisheye undistort, or estimate a new K from balance/fov-scale.",
    )
    parser.add_argument("--crop", action="store_true", help="Crop standard model output to valid ROI.")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--visual-scale", type=float, default=0.35)
    return parser


def main():
    args = build_parser().parse_args()
    params = load_params(args.params)
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise SystemExit(f"No input images found in {args.input_dir}")

    output_dirs = make_output_dirs(args.output_dir)
    saved_count = 0
    for image_path in image_paths:
        output_paths = process_image(
            image_path,
            output_dirs,
            params,
            args.model,
            args.alpha,
            args.balance,
            args.fov_scale,
            args.fisheye_new_camera,
            args.crop,
            not args.no_comparison,
            args.visual_scale,
        )
        saved_count += len(output_paths)
        for output_path in output_paths:
            print(f"Saved: {output_path}")

    print(f"Processed {len(image_paths)} images. Saved {saved_count} files.")


if __name__ == "__main__":
    main()
