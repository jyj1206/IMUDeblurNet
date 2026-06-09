#!/usr/bin/env bash
set -e

python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/GoPro \
  --split test \
  --allow-missing-gt \
  --camera-fx 960.0 \
  --camera-fy 960.0 \
  --camera-cx 640.0 \
  --camera-cy 360.0
