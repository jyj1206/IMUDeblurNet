#!/usr/bin/env bash
set -e

if [ ! -d "data/IMURealBlur/test/camera_motion_field" ] || ! find "data/IMURealBlur/test/camera_motion_field" -type f \( -name "*.npy" -o -name "*.npz" \) -print -quit | grep -q .; then
  python generate_camera_motion_field.py \
    --data_root data/IMURealBlur \
    --mode test \
    --overwrite
fi

python validate_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
