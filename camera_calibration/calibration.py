import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT_DIR = Path("data") / "checkerboard_image"
DEFAULT_OUTPUT_DIR = Path("result") / "calibration"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")


def parse_board_size(value):
    cleaned = value.lower().replace(",", "x")
    parts = cleaned.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Board size must look like 8x6.")

    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Board size values must be integers.") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Board size values must be positive.")
    return width, height


def collect_images(image_dir):
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(pattern))
        paths.extend(image_dir.glob(pattern.upper()))
    return sorted(set(paths))


def make_object_points(board_size, square_size):
    width, height = board_size
    grid = np.mgrid[0:width, 0:height].T.reshape(-1, 2).astype(np.float32)
    grid *= square_size

    objp_standard = np.zeros((1, width * height, 3), np.float32)
    objp_standard[0, :, :2] = grid

    objp_fisheye = np.zeros((width * height, 1, 3), np.float32)
    objp_fisheye[:, 0, :2] = grid
    return objp_standard, objp_fisheye


def find_calibration_points(image_paths, board_size, square_size):
    objp_standard, objp_fisheye = make_object_points(board_size, square_size)
    objpoints_standard = []
    objpoints_fisheye = []
    imgpoints = []
    used_paths = []
    skipped_paths = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_FAST_CHECK
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            skipped_paths.append(image_path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_size = gray.shape[::-1]
        if image_size is None:
            image_size = current_size
        elif image_size != current_size:
            skipped_paths.append(image_path)
            continue

        found, corners = cv2.findChessboardCorners(gray, board_size, flags)
        if not found:
            skipped_paths.append(image_path)
            continue

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints_standard.append(objp_standard)
        objpoints_fisheye.append(objp_fisheye)
        imgpoints.append(refined)
        used_paths.append(image_path)

    if image_size is None:
        raise RuntimeError("No readable calibration images were found.")
    if not imgpoints:
        raise RuntimeError("No checkerboard corners were detected. Check --board-size.")

    return objpoints_standard, objpoints_fisheye, imgpoints, image_size, used_paths, skipped_paths


def calibrate_standard(objpoints, imgpoints, image_size):
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    return rms, camera_matrix, dist_coeffs


def calibrate_fisheye(objpoints, imgpoints, image_size):
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        rms, camera_matrix, dist_coeffs, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            image_size,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            flags,
            criteria,
        )
    except cv2.error:
        flags -= cv2.fisheye.CALIB_CHECK_COND
        rms, camera_matrix, dist_coeffs, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            image_size,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs,
            flags,
            criteria,
        )

    return rms, camera_matrix, dist_coeffs


def undistort_standard(image, camera_matrix, dist_coeffs, alpha):
    height, width = image.shape[:2]
    image_size = (width, height)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha,
        image_size,
    )
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)


def undistort_fisheye(image, camera_matrix, dist_coeffs, balance, fov_scale, new_camera_mode):
    height, width = image.shape[:2]
    image_size = (width, height)
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
    return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)


def draw_label(image, label, color):
    output = image.copy()
    cv2.putText(
        output,
        label,
        (18, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        color,
        3,
        cv2.LINE_AA,
    )
    return output


def save_comparison(
    sample_path,
    output_path,
    standard_matrix,
    standard_dist,
    fisheye_matrix,
    fisheye_dist,
    alpha,
    balance,
    fov_scale,
    fisheye_new_camera,
    visual_scale,
):
    image = cv2.imread(str(sample_path))
    if image is None:
        return

    standard = undistort_standard(image, standard_matrix, standard_dist, alpha)
    fisheye = undistort_fisheye(
        image,
        fisheye_matrix,
        fisheye_dist,
        balance,
        fov_scale,
        fisheye_new_camera,
    )

    views = [
        draw_label(image, "Original", (0, 0, 255)),
        draw_label(standard, "Standard alpha=%.2f" % alpha, (0, 180, 0)),
        draw_label(fisheye, "Fisheye balance=%.2f" % balance, (255, 0, 0)),
    ]
    if visual_scale != 1.0:
        height, width = image.shape[:2]
        target_size = (int(width * visual_scale), int(height * visual_scale))
        views = [cv2.resize(view, target_size) for view in views]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.hstack(views))


def write_report(
    output_path,
    input_dir,
    image_size,
    board_size,
    square_size,
    used_paths,
    skipped_paths,
    rms_standard,
    matrix_standard,
    dist_standard,
    rms_fisheye,
    matrix_fisheye,
    dist_fisheye,
):
    lines = [
        "=" * 50,
        "CAMERA CALIBRATION PARAMETERS REPORT",
        "=" * 50,
        f"Target Directory : {input_dir}",
        f"Image Dimensions : {image_size[0]} x {image_size[1]} (W x H)",
        f"Checkerboard     : {board_size[0]} x {board_size[1]} inner corners",
        f"Square Size      : {square_size}",
        f"Used Images Count: {len(used_paths)} frames",
        f"Skipped Count    : {len(skipped_paths)} frames",
        "=" * 50,
        "",
        "--- [METHOD 1] STANDARD CAMERA MODEL ---",
        f"Re-projection Error (RMS): {rms_standard:.6f} pixels",
        "",
        "Camera Matrix (K):",
        np.array2string(matrix_standard, separator=", "),
        "",
        "Distortion Coefficients ([k1, k2, p1, p2, k3, ...]):",
        np.array2string(dist_standard.flatten(), separator=", "),
        "",
        "=" * 50,
        "",
        "--- [METHOD 2] FISHEYE CAMERA MODEL ---",
        f"Re-projection Error (RMS): {rms_fisheye:.6f} pixels",
        "",
        "Camera Matrix (K):",
        np.array2string(matrix_fisheye, separator=", "),
        "",
        "Distortion Coefficients ([k1, k2, k3, k4]):",
        np.array2string(dist_fisheye.flatten(), separator=", "),
        "",
        "=" * 50,
        "",
        "Used Images:",
    ]
    lines.extend([f"- {path.name}" for path in used_paths])
    if skipped_paths:
        lines.extend(["", "Skipped Images:"])
        lines.extend([f"- {path.name}" for path in skipped_paths])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate standard and fisheye camera models from checkerboard images.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--board-size", type=parse_board_size, default=(8, 6))
    parser.add_argument("--square-size", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.0, help="Standard undistort free scaling, 0 crops black borders and 1 keeps all pixels.")
    parser.add_argument("--balance", type=float, default=0.0, help="Fisheye balance, 0 crops black borders and 1 keeps wider FOV.")
    parser.add_argument("--fov-scale", type=float, default=1.0)
    parser.add_argument(
        "--fisheye-new-camera",
        choices=("same", "estimate"),
        default="same",
        help="Use the original K as fisheye new camera matrix, or estimate one from balance/fov-scale.",
    )
    parser.add_argument("--visual-scale", type=float, default=0.5)
    parser.add_argument("--params-name", default="calibration_params.npz")
    parser.add_argument("--report-name", default="calibration_results.txt")
    parser.add_argument("--comparison-name", default="combined_calibration_comparison.jpg")
    return parser


def main():
    args = build_parser().parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No calibration images found in {input_dir}")

    (
        objpoints_standard,
        objpoints_fisheye,
        imgpoints,
        image_size,
        used_paths,
        skipped_paths,
    ) = find_calibration_points(image_paths, args.board_size, args.square_size)

    rms_standard, matrix_standard, dist_standard = calibrate_standard(
        objpoints_standard,
        imgpoints,
        image_size,
    )
    rms_fisheye, matrix_fisheye, dist_fisheye = calibrate_fisheye(
        objpoints_fisheye,
        imgpoints,
        image_size,
    )

    params_path = output_dir / args.params_name
    np.savez(
        params_path,
        image_width=image_size[0],
        image_height=image_size[1],
        board_width=args.board_size[0],
        board_height=args.board_size[1],
        square_size=args.square_size,
        rms_standard=rms_standard,
        camera_matrix_standard=matrix_standard,
        dist_coeffs_standard=dist_standard,
        rms_fisheye=rms_fisheye,
        camera_matrix_fisheye=matrix_fisheye,
        dist_coeffs_fisheye=dist_fisheye,
    )

    report_path = output_dir / args.report_name
    comparison_path = output_dir / args.comparison_name
    write_report(
        report_path,
        input_dir,
        image_size,
        args.board_size,
        args.square_size,
        used_paths,
        skipped_paths,
        rms_standard,
        matrix_standard,
        dist_standard,
        rms_fisheye,
        matrix_fisheye,
        dist_fisheye,
    )
    save_comparison(
        used_paths[0],
        comparison_path,
        matrix_standard,
        dist_standard,
        matrix_fisheye,
        dist_fisheye,
        args.alpha,
        args.balance,
        args.fov_scale,
        args.fisheye_new_camera,
        args.visual_scale,
    )

    print(f"Used {len(used_paths)} calibration images; skipped {len(skipped_paths)}.")
    print(f"Saved parameters: {params_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
