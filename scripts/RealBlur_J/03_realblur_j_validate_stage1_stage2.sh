#!/usr/bin/env bash
set -e

python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/RealBlur_J \
  --split test \
  --realblur-metrics \
  --camera-fx 1000.0 \
  --camera-fy 1000.0 \
  --camera-cx 344.0 \
  --camera-cy 392.0
