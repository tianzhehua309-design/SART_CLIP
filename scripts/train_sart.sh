#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
CKPT_ROOT="${CKPT_ROOT:-checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TRAIN_PATH="${TRAIN_PATH:-${DATA_ROOT}/ImageNet/train}"
SEM_DESC_JSON="${SEM_DESC_JSON:-prompt/sem_desc_all_datasets_fg_bias_hybrid.json}"

python model/sart_clip.py \
  --train-path "${TRAIN_PATH}" \
  --ckpt-root "${CKPT_ROOT}" \
  --sem-desc-json "${SEM_DESC_JSON}" \
  --output-dir "${OUTPUT_ROOT}/sart_ablation_runs" \
  "$@"
