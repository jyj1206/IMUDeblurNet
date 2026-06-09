#!/usr/bin/env bash
set -e

if [ ! -d "data/IMUBlur/test/camera_motion_field" ] || ! find "data/IMUBlur/test/camera_motion_field" -type f \( -name "*.npy" -o -name "*.npz" \) -print -quit | grep -q .; then
  python generate_camera_motion_field.py \
    --data_root data/IMUBlur \
    --mode test \
    --overwrite
fi

python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMUBlur \
  --split test \
  --load-target-gyro
