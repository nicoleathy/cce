#!/bin/bash
# Robustness checks (Section 4.5): ARC-Challenge and WinoGrande, 0-shot, orig + exhaustive.
#
# Usage:
#   bash scripts/run_robustness.sh <chat|fp32|wording> [output-root]
#
#   chat     native chat templates, bfloat16 (four instruction-tuned models)
#   fp32     plain prompt, float32 scoring (same four models)
#   wording  NOTA replaced by "None of these options is correct" (three models)
#
# Everything else matches scripts/run_cce.sh, so only the factor under test changes.
# Afterwards:
#   python analysis/paired_analysis.py --root <output-root> --out <mode>.csv
#   python analysis/compare_runs.py --base results/cce_tables.csv --new <mode>.csv
#
# Chat mode and Qwen3.5: the Qwen3.5 chat template opens a <think> block. Run
# `python scripts/make_qwen_nothink_tok.py --out <dir>` once and set
# QWEN_NOTHINK_TOK=<dir> so the block is closed before the answer letter.
#
# Environment variables: MODELS_OVERRIDE (space-separated model ids), REVISION
# (pin a commit; use with a single model in MODELS_OVERRIDE), BATCH_SIZE,
# PARALLELIZE, PARALLELIZE_FP32, DTYPE_FP32, LIMIT, LAUNCH (see run_cce.sh).

set -uo pipefail

MODE="${1:?usage: run_robustness.sh <chat|fp32|wording> [output-root]}"
OUTROOT="${2:-outputs/robustness/${MODE}}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO}/tasks/cce_mcf"

LIMIT="${LIMIT:-}"
LAUNCH="${LAUNCH:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARALLELIZE="${PARALLELIZE:-False}"
DTYPE="bfloat16"
EXTRA=()
IT_MODELS="meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen3.5-4B google/gemma-4-12B-it"
QWEN_NOTHINK_TOK="${QWEN_NOTHINK_TOK:-}"
REVISION="${REVISION:-}"

case "${MODE}" in
  chat)
    MODELS="${MODELS_OVERRIDE:-${IT_MODELS}}"
    EXTRA+=(--apply_chat_template) ;;
  fp32)
    MODELS="${MODELS_OVERRIDE:-${IT_MODELS}}"
    DTYPE="${DTYPE_FP32:-float32}"
    PARALLELIZE="${PARALLELIZE_FP32:-True}" ;;  # float32 doubles memory
  wording)
    MODELS="${MODELS_OVERRIDE:-meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-v0.3 google/gemma-4-31B}"
    export CCE_NOTA="None of these options is correct" ;;  # read by tasks/cce_mcf/utils.py
  *) echo "unknown mode ${MODE}; use chat, fp32 or wording"; exit 1 ;;
esac
[[ -n "${LIMIT}" ]] && EXTRA+=(--limit "${LIMIT}")

export CCE_FEWSHOT_SEED=1234
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTROOT}"
MANIFEST="${OUTROOT}/run_manifest.txt"
{
  echo "=== $(date -Iseconds) ==="
  echo "mode=${MODE} revision=${REVISION:-latest} dtype=${DTYPE} batch_size=${BATCH_SIZE} parallelize=${PARALLELIZE} limit=${LIMIT:-none} nota=${CCE_NOTA:-default}"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
} 2>&1 | tee -a "${MANIFEST}"

FAILED=()
for MODEL in ${MODELS}; do
  MARGS="pretrained=${MODEL},parallelize=${PARALLELIZE},attn_implementation=sdpa,dtype=${DTYPE}"
  [[ -n "${REVISION}" ]] && MARGS+=",revision=${REVISION}"
  if [[ "${MODE}" == chat && "${MODEL}" == Qwen/* ]]; then
    if [[ -z "${QWEN_NOTHINK_TOK}" ]]; then
      echo "!!! set QWEN_NOTHINK_TOK (see header) before running ${MODEL} in chat mode"
      FAILED+=("${MODEL}"); continue
    fi
    MARGS+=",tokenizer=${QWEN_NOTHINK_TOK}"
  fi
  for KIND in orig exh; do
    echo "=== ${MODE} ${MODEL} ${KIND} ==="
    if ! CCE_SEED=0 ${LAUNCH} python3 -m lm_eval \
        --model hf \
        --model_args "${MARGS}" \
        --include_path "${TASK_DIR}" \
        --tasks "arc_challenge_mcf_${KIND},winogrande_mcf_${KIND}" \
        --device cuda --batch_size "${BATCH_SIZE}" --num_fewshot 0 \
        --seed 0,0,0,1234 \
        --output_path "${OUTROOT}/${MODEL}/${KIND}_0shot" --log_samples \
        "${EXTRA[@]+"${EXTRA[@]}"}"; then
      echo "!!! ${MODEL} ${KIND} FAILED"; FAILED+=("${MODEL}/${KIND}")
    fi
    echo "${MODEL} ${KIND} finished $(date -Iseconds)" >> "${MANIFEST}"
  done
done

if (( ${#FAILED[@]} )); then echo "Finished with failures: ${FAILED[*]}"; exit 1; fi
echo "All runs finished. Results in ${OUTROOT}"
