#!/usr/bin/env bash
set -euo pipefail

# Repo root (parent of configs/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Florence-2-large + bf16 on 16GB A4000: start at 4 images per GPU; if CUDA OOM, try 2 or 1.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"

# Optional: export TRAIN_PHASE=phase2 for honest training (loads phase1 checkpoint by default).
TRAIN_PHASE="${TRAIN_PHASE:-phase1}"

accelerate launch \
    --num_processes=4 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip="10.69.180.212" \
    --main_process_port=29500 \
    train.py \
    --phase "${TRAIN_PHASE}" \
    --coco_split train \
    --batch_size "${PER_DEVICE_BATCH}" \
    --epochs 10 \
    --num_samples 2000 \
    --output_dir ./checkpoints \
    --no_eval_baseline
