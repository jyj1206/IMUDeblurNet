#!/usr/bin/env bash
set -e

python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/RealBlur_J \
  --split test \
  --allow-missing-gt \
  --realblur-metrics \
  --camera-fx 1000.0 \
  --camera-fy 1000.0 \
  --camera-cx 344.0 \
  --camera-cy 392.0
