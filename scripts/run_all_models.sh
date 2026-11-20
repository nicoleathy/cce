#!/bin/bash
# Run scripts/run_cce.sh for every model in scripts/models.tsv at the
# checkpoint used in the paper.
#
#   bash scripts/run_all_models.sh results/runs
#
# The larger models (Qwen3.5-27B, Gemma-4 12B/31B) need several GPUs:
#   PARALLELIZE=True BATCH_SIZE=2 bash scripts/run_all_models.sh results/runs

set -uo pipefail
OUTROOT="${1:?usage: run_all_models.sh <output-root>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -v '^#' "${REPO_ROOT}/scripts/models.tsv" | while IFS=$'\t' read -r MODEL REV _NAME; do
  [[ -z "${MODEL}" ]] && continue
  bash "${REPO_ROOT}/scripts/run_cce.sh" "${MODEL}" "${OUTROOT}" "${REV}"
done
