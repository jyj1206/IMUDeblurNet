#!/usr/bin/env bash
set -e

python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
