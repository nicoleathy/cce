#!/bin/bash
# Cloze-scoring CCE runs (Section 4.4, Table 3).
#
# Usage:
#   bash scripts/run_cloze.sh <hf-model-id> [output-root]
#
# Each option is scored by the summed log-likelihood of its full text. One
# random option per item is replaced by "None of the above" in place (option
# order matters little under cloze scoring, since each option is scored as a
# separate continuation). Afterwards:
#   python analysis/length_artifact.py --root <output-root> --out length_artifact.csv
#
# Environment variables: SEEDS (default "0"; Table 3 uses seed 0), BATCH_SIZE
# (default auto), PARALLELIZE (default True), LIMIT, LAUNCH (see run_cce.sh).

set -uo pipefail

MODEL="${1:?usage: run_cloze.sh <hf-model-id> [output-root]}"
OUTROOT="${2:-outputs/cloze}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO}/tasks/cce_cloze"
OUTDIR="${OUTROOT}/${MODEL}"

SEEDS="${SEEDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
PARALLELIZE="${PARALLELIZE:-True}"
LIMIT="${LIMIT:-}"
LAUNCH="${LAUNCH:-}"
TASKS="arc_challenge_cce_cloze,commonsense_qa_cce_cloze,hellaswag_cce_cloze,piqa_cce_cloze,winogrande_cce_cloze"

mkdir -p "${OUTDIR}"
FAILED=()
for s in ${SEEDS}; do
  extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  echo "=== cloze seed ${s} ==="
  if ! CCE_SEED="${s}" ${LAUNCH} python3 -m lm_eval \
      --model hf \
      --model_args "pretrained=${MODEL},parallelize=${PARALLELIZE},attn_implementation=sdpa" \
      --include_path "${TASK_DIR}" \
      --tasks "${TASKS}" \
      --device cuda --batch_size "${BATCH_SIZE}" --num_fewshot 0 --seed "${s}" \
      --output_path "${OUTDIR}/cce_0shot_seed${s}" --log_samples \
      "${extra[@]+"${extra[@]}"}"; then
    echo "!!! seed ${s} FAILED"; FAILED+=("${s}")
  fi
done

if (( ${#FAILED[@]} )); then echo "Finished with failures: seeds ${FAILED[*]}"; exit 1; fi
echo "All runs finished. Results in ${OUTDIR}"
