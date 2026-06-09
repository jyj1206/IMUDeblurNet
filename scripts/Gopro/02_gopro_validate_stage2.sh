#!/usr/bin/env bash
set -e

if [ ! -d "data/GoPro/test/camera_motion_field" ] || ! find "data/GoPro/test/camera_motion_field" -type f \( -name "*.npy" -o -name "*.npz" \) -print -quit | grep -q .; then
  python generate_camera_motion_field.py \
    --data_root data/GoPro \
    --mode test \
    --camera-fx 960.0 \
    --camera-fy 960.0 \
    --camera-cx 640.0 \
    --camera-cy 360.0 \
    --overwrite
fi

python validate_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --dataset-root data/GoPro \
  --split test
