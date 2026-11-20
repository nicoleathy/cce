#!/bin/bash
# Cloze-scored CCE for the length-artifact analysis (Section 4.4, Table 3).
#
# Usage (from the repository root):
#   bash scripts/run_cloze.sh <hf-model-id> <output-root>
#
# Table 3 uses Llama-3.2-3B, Llama-3.2-3B-Instruct, Mistral-7B-v0.3 and
# Mistral-7B-Instruct-v0.3, 0-shot, substitution seed 0. Afterwards:
#   python analysis/length_artifact.py --root <output-root> --seed 0 --out length_artifact.csv
#   python analysis/build_table3.py --csv length_artifact.csv
#
# Optional: SEEDS="0" (other seeds are supported), BATCH_SIZE, PARALLELIZE,
# LAUNCH, LIMIT, REVISION, as in run_cce.sh.

set -uo pipefail
MODEL="${1:?usage: run_cloze.sh <hf-model-id> <output-root>}"
OUTROOT="${2:?usage: run_cloze.sh <hf-model-id> <output-root>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-0}"
REVISION="${REVISION:-main}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARALLELIZE="${PARALLELIZE:-False}"
LAUNCH="${LAUNCH:-}"
LIMIT="${LIMIT:-}"
EXTRA=()
[[ -n "${LIMIT}" ]] && EXTRA+=(--limit "${LIMIT}")

TASKS="arc_challenge_cloze_cce,commonsense_qa_cloze_cce,hellaswag_cloze_cce,piqa_cloze_cce,winogrande_cloze_cce"
for s in ${SEEDS}; do
  echo "=== ${MODEL} cloze CCE, CCE_SEED=${s} ==="
  # shellcheck disable=SC2086
  CCE_SEED="${s}" ${LAUNCH} python3 -m lm_eval \
    --model hf \
    --model_args "pretrained=${MODEL},revision=${REVISION},parallelize=${PARALLELIZE},attn_implementation=sdpa" \
    --include_path "${REPO_ROOT}/tasks/cce_cloze" \
    --tasks "${TASKS}" \
    --device cuda --batch_size "${BATCH_SIZE}" --num_fewshot 0 \
    --seed "${s},${s},${s},1234" \
    --output_path "${OUTROOT}/${MODEL}/cce_0shot_seed${s}" \
    --log_samples \
    "${EXTRA[@]+"${EXTRA[@]}"}" || echo "!!! ${MODEL} seed ${s} FAILED"
done
