#!/usr/bin/env bash
set -e

python validate_stage1.py \
  --checkpoint weights/best_stage1.pt \
  --dataset-root data/IMUBlurV2 \
  --split test
