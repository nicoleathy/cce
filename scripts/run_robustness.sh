#!/bin/bash
# Robustness checks (Sections 4.3 and 4.5): 0-shot Orig and exhaustive CCE on
# ARC and WinoGrande, changing one factor at a time.
#
# Usage (from the repository root):
#   bash scripts/run_robustness.sh chat    results/robustness_runs/chat     # native chat template
#   bash scripts/run_robustness.sh fp32    results/robustness_runs/fp32     # float32 instead of bfloat16
#   bash scripts/run_robustness.sh wording results/robustness_runs/wording  # "None of these options is correct"
#   bash scripts/run_robustness.sh nobos   results/robustness_runs/no_bos   # no BOS token (Llama-3.2-1B)
#
# Then compare with the main results:
#   python analysis/paired_analysis.py --root results/robustness_runs/chat --out cce_chat.csv
#   python analysis/compare_runs.py --base results/tables/cce_tables_final.csv --new cce_chat.csv
#
# For the chat mode, Qwen3.5's chat template opens a <think> block. Create a
# tokenizer copy whose template closes it first:
#   python scripts/make_qwen_nothink_tok.py --out qwen35_4b_nothink_tok
# and point QWEN_NOTHINK_TOK at that folder (default: ./qwen35_4b_nothink_tok).
#
# Same optional variables as run_cce.sh (BATCH_SIZE, PARALLELIZE, LAUNCH, LIMIT).
# MODELS_OVERRIDE="org/model ..." replaces the default model list.

set -uo pipefail
MODE="${1:?usage: run_robustness.sh <chat|fp32|wording|nobos> <output-root>}"
OUTROOT="${2:?usage: run_robustness.sh <chat|fp32|wording|nobos> <output-root>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_NOTHINK_TOK="${QWEN_NOTHINK_TOK:-${PWD}/qwen35_4b_nothink_tok}"

IT_MODELS="meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen3.5-4B google/gemma-4-12B-it"

case "${MODE}" in
  chat)    MODELS="${MODELS_OVERRIDE:-${IT_MODELS}}"
           export EXTRA_LM_EVAL_ARGS="--apply_chat_template" ;;
  fp32)    MODELS="${MODELS_OVERRIDE:-${IT_MODELS}}"
           export DTYPE=float32 ;;
  wording) MODELS="${MODELS_OVERRIDE:-meta-llama/Llama-3.2-3B-Instruct mistralai/Mistral-7B-v0.3 google/gemma-4-31B}"
           export CCE_NOTA="None of these options is correct" ;;
  nobos)   MODELS="${MODELS_OVERRIDE:-meta-llama/Llama-3.2-1B}"
           export ADD_BOS=False ;;
  *) echo "unknown mode ${MODE}; use chat, fp32, wording or nobos"; exit 1 ;;
esac

# Only the 0-shot Orig and exhaustive runs, only ARC and WinoGrande.
export SHOTS_LIST=0 RUN_CCE=0 DATASETS="arc_challenge winogrande"
FAILED=()
for MODEL in ${MODELS}; do
  EXTRA_ARGS_SAVED="${EXTRA_MODEL_ARGS:-}"
  if [[ "${MODE}" == chat && "${MODEL}" == Qwen/* ]]; then
    export EXTRA_MODEL_ARGS="${EXTRA_ARGS_SAVED},tokenizer=${QWEN_NOTHINK_TOK}"
  fi
  bash "${REPO_ROOT}/scripts/run_cce.sh" "${MODEL}" "${OUTROOT}" \
    || FAILED+=("${MODEL}")
  export EXTRA_MODEL_ARGS="${EXTRA_ARGS_SAVED}"
done

if (( ${#FAILED[@]} )); then echo "Finished with failures: ${FAILED[*]}"; exit 1; fi
echo "All ${MODE} runs finished. Results in ${OUTROOT}"
