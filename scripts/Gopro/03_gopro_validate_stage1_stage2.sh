#!/usr/bin/env bash
set -e

python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/GoPro \
  --split test \
  --camera-fx 960.0 \
  --camera-fy 960.0 \
  --camera-cx 640.0 \
  --camera-cy 360.0
