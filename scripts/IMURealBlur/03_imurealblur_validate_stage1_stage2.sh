#!/usr/bin/env bash
set -e

python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
