#!/bin/bash
# Main CCE runs (Tables 1 and 2) for one model, letter scoring.
#
# Usage (from the repository root):
#   bash scripts/run_cce.sh <hf-model-id> <output-root> [revision]
#
# Example:
#   bash scripts/run_cce.sh meta-llama/Llama-3.2-3B results/runs \
#        13afe5124825b4f3751f836b40dafda64c1ed062
#
# Writes, under <output-root>/<hf-model-id>/:
#   orig_{S}shot          original items (no substitution)
#   cce_{S}shot_seed{N}   one random removal per item, substitution seeds N
#   exh_0shot             exhaustive design: every item x every removed option
# This is the layout analysis/paired_analysis.py expects.
#
# Optional environment variables:
#   SHOTS_LIST="0 5"       shot settings for orig and cce
#   SEEDS="0 1 2 3 4"      substitution seeds
#   DATASETS="arc_challenge ..."  subset of the five benchmarks
#   FEWSHOT_SEED=1234      fixes which demonstrations are drawn and how they are
#                          substituted, so every run sees the same demonstrations
#   RUN_ORIG=1 RUN_CCE=1 RUN_EXH=1   set any to 0 to skip that family
#   DTYPE=bfloat16         model dtype (the paper uses bfloat16)
#   BATCH_SIZE=8           lm-eval batch size
#   PARALLELIZE=False      True to shard a large model over several GPUs
#   ADD_BOS=True           prepend the BOS token (the paper does, for models that define one)
#   EXTRA_MODEL_ARGS=""    appended to --model_args, e.g. ",tokenizer=<path>"
#   EXTRA_LM_EVAL_ARGS=""  appended to the lm_eval call, e.g. "--apply_chat_template"
#   LAUNCH=""              prefix for the python call, e.g. "srun --gres=gpu:4"
#   LIMIT=20               pass --limit for a quick smoke test
#
# Gated models (Llama, Gemma) need a Hugging Face login: run `hf auth login`
# once, or export HF_TOKEN in your shell. Never write a token into this script.

set -uo pipefail

MODEL="${1:?usage: run_cce.sh <hf-model-id> <output-root> [revision]}"
OUTROOT="${2:?usage: run_cce.sh <hf-model-id> <output-root> [revision]}"
REVISION="${3:-main}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO_ROOT}/tasks/cce_mcf"
OUTDIR="${OUTROOT}/${MODEL}"

SHOTS_LIST="${SHOTS_LIST:-0 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FEWSHOT_SEED="${FEWSHOT_SEED:-1234}"
RUN_ORIG="${RUN_ORIG:-1}"
RUN_CCE="${RUN_CCE:-1}"
RUN_EXH="${RUN_EXH:-1}"
DTYPE="${DTYPE:-bfloat16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARALLELIZE="${PARALLELIZE:-False}"
ADD_BOS="${ADD_BOS:-True}"
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
EXTRA_LM_EVAL_ARGS="${EXTRA_LM_EVAL_ARGS:-}"
LAUNCH="${LAUNCH:-}"
LIMIT="${LIMIT:-}"

DATASETS="${DATASETS:-arc_challenge commonsense_qa hellaswag piqa winogrande}"
tasks_for () {  # tasks_for <orig|cce|exh>
  local out=""
  for d in ${DATASETS}; do out="${out:+${out},}${d}_mcf_$1"; done
  echo "${out}"
}

MODEL_ARGS="pretrained=${MODEL},revision=${REVISION},dtype=${DTYPE},parallelize=${PARALLELIZE},attn_implementation=sdpa,add_bos_token=${ADD_BOS}${EXTRA_MODEL_ARGS}"

export CCE_FEWSHOT_SEED="${FEWSHOT_SEED}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/run_manifest.txt"
{
  echo "=== run $(date +%Y-%m-%dT%H:%M:%S) ==="
  echo "model=${MODEL} revision=${REVISION}"
  echo "shots=${SHOTS_LIST} seeds=${SEEDS} fewshot_seed=${FEWSHOT_SEED} limit=${LIMIT:-none}"
  echo "batch_size=${BATCH_SIZE} parallelize=${PARALLELIZE} dtype=${DTYPE}"
  python3 -c "import transformers, torch, lm_eval; print('transformers', transformers.__version__, '| torch', torch.__version__, '| lm_eval', getattr(lm_eval, '__version__', 'unknown'))"
} 2>&1 | tee -a "${LOG}"

FAILED=()

run () {  # run <tasks> <num_fewshot> <substitution seed> <run name>
  local extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  echo "=== $4 (shots=$2, CCE_SEED=$3) ==="
  # shellcheck disable=SC2086
  if ! CCE_SEED="$3" ${LAUNCH} python3 -m lm_eval \
      --model hf \
      --model_args "${MODEL_ARGS}" \
      --include_path "${TASK_DIR}" \
      --tasks "$1" \
      --device cuda --batch_size "${BATCH_SIZE}" \
      --num_fewshot "$2" \
      --seed "$3,$3,$3,${FEWSHOT_SEED}" \
      --output_path "${OUTDIR}/$4" \
      --log_samples \
      ${EXTRA_LM_EVAL_ARGS} \
      "${extra[@]+"${extra[@]}"}"; then
    echo "!!! $4 FAILED"
    FAILED+=("$4")
  fi
  echo "$4 finished $(date +%Y-%m-%dT%H:%M:%S)" >> "${LOG}"
}

for SHOTS in ${SHOTS_LIST}; do
  [[ "${RUN_ORIG}" == 1 ]] && run "$(tasks_for orig)" "${SHOTS}" 0 "orig_${SHOTS}shot"
  if [[ "${RUN_CCE}" == 1 ]]; then
    for s in ${SEEDS}; do
      run "$(tasks_for cce)" "${SHOTS}" "${s}" "cce_${SHOTS}shot_seed${s}"
    done
  fi
done

# Exhaustive design: 0-shot only (the substitution seed has no effect).
[[ "${RUN_EXH}" == 1 ]] && run "$(tasks_for exh)" 0 0 "exh_0shot"

if (( ${#FAILED[@]} )); then
  echo "Finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "All runs finished. Results in ${OUTDIR}"
