#!/usr/bin/env bash
set -e

if [ ! -d "data/RealBlur_J/test/camera_motion_field" ] || ! find "data/RealBlur_J/test/camera_motion_field" -type f \( -name "*.npy" -o -name "*.npz" \) -print -quit | grep -q .; then
  python generate_camera_motion_field.py \
    --data_root data/RealBlur_J \
    --mode test \
    --camera-fx 1000.0 \
    --camera-fy 1000.0 \
    --camera-cx 344.0 \
    --camera-cy 392.0 \
    --overwrite
fi

python validate_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --dataset-root data/RealBlur_J \
  --split test \
  --realblur-metrics
