#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/path/to/datasets}"
CKPT_ROOT="${CKPT_ROOT:-checkpoint}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
SEM_DESC_JSON="${SEM_DESC_JSON:-prompt/sem_desc_all_datasets_fg_bias_hybrid.json}"

python eval/eval_autoattack.py \
  --data-root "${DATA_ROOT}" \
  --ckpt-root "${CKPT_ROOT}" \
  --sem-desc-json "${SEM_DESC_JSON}" \
  --methods clip_vit_b32 sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --results-csv "${OUTPUT_ROOT}/evaluation/aa_fast_results.csv" \
  "$@"
