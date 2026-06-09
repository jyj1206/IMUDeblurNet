#!/usr/bin/env bash
set -e

if [ ! -d "data/IMUBlurV2/test/camera_motion_field" ] || ! find "data/IMUBlurV2/test/camera_motion_field" -type f \( -name "*.npy" -o -name "*.npz" \) -print -quit | grep -q .; then
  python generate_camera_motion_field.py \
    --data_root data/IMUBlurV2 \
    --mode test \
    --overwrite
fi

python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/IMUBlurV2 \
  --split test \
  --load-target-gyro
