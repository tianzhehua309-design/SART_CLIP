#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
TRAIN_PATH="${TRAIN_PATH:-${DATA_ROOT}/ImageNet/train}"
SEM_DESC_JSON="${SEM_DESC_JSON:-prompt/sem_desc_imagenet_fg_bias.json}"

python model/sp_casa.py \
  --train-path "${TRAIN_PATH}" \
  --sem-desc-json "${SEM_DESC_JSON}" \
  --output-dir "${OUTPUT_ROOT}/sp_casa_ablation_runs" \
  "$@"
