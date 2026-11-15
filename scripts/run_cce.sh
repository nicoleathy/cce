#!/bin/bash
# Main CCE evaluation for one model (letter scoring; Tables 1 and 2).
#
# Usage:
#   bash scripts/run_cce.sh <hf-model-id> [output-root]
#
# Example:
#   bash scripts/run_cce.sh meta-llama/Llama-3.2-3B outputs/main
#
# Runs, for every benchmark (ARC-Challenge, CommonsenseQA, HellaSwag, PIQA,
# WinoGrande):
#   orig_{S}shot          original items, no substitution
#   cce_{S}shot_seed{N}   one random removal per item, substitution seed N
#   exh_0shot             exhaustive design: every item x every removed option
#
# Results go to <output-root>/<hf-model-id>/<run>/. Then run
#   python analysis/paired_analysis.py --root <output-root> --out cce_tables.csv
#
# Environment variables (defaults are the settings used in the paper):
#   SHOTS_LIST="0 5"       shot settings for orig and cce runs
#   SEEDS="0 1 2 3 4"      substitution seeds
#   FEWSHOT_SEED=1234      fixed seed for demonstration sampling and for the
#                          substitution applied to demonstrations
#   BATCH_SIZE=8           lower it (4 or 2) for larger models
#   PARALLELIZE=False      True splits the model across all visible GPUs
#   DTYPE=bfloat16
#   REVISION=              pin a Hugging Face commit (see results/checkpoints.csv)
#   RUN_ORIG=1 RUN_CCE=1 RUN_EXH=1   set any to 0 to skip that family
#   LIMIT=                 e.g. LIMIT=20 for a quick smoke test
#   LAUNCH=                command prefix, e.g. LAUNCH="srun --gres=gpu:4"
#
# Hugging Face access: run `hf auth login` once (or export HF_TOKEN).
# Never write a token into this file.

set -uo pipefail

MODEL="${1:?usage: run_cce.sh <hf-model-id> [output-root]}"
OUTROOT="${2:-outputs/main}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_DIR="${REPO}/tasks/cce_mcf"
OUTDIR="${OUTROOT}/${MODEL}"

SHOTS_LIST="${SHOTS_LIST:-0 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FEWSHOT_SEED="${FEWSHOT_SEED:-1234}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARALLELIZE="${PARALLELIZE:-False}"
DTYPE="${DTYPE:-bfloat16}"
REVISION="${REVISION:-}"
RUN_ORIG="${RUN_ORIG:-1}"
RUN_CCE="${RUN_CCE:-1}"
RUN_EXH="${RUN_EXH:-1}"
LIMIT="${LIMIT:-}"
LAUNCH="${LAUNCH:-}"

DATASETS="arc_challenge commonsense_qa hellaswag piqa winogrande"
tasks_for () {  # tasks_for <orig|cce|exh>
  local out=""
  for d in ${DATASETS}; do out="${out:+${out},}${d}_mcf_$1"; done
  echo "${out}"
}

# Read by tasks/cce_mcf/utils.py: substitution seed for the demonstration pool,
# fixed so that every run sees the same demonstrations.
export CCE_FEWSHOT_SEED="${FEWSHOT_SEED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUTDIR}"
MANIFEST="${OUTDIR}/run_manifest.txt"
{
  echo "=== $(date -Iseconds) ==="
  echo "model=${MODEL}"
  echo "shots=${SHOTS_LIST} seeds=${SEEDS} fewshot_seed=${FEWSHOT_SEED} limit=${LIMIT:-none}"
  echo "batch_size=${BATCH_SIZE} parallelize=${PARALLELIZE} dtype=${DTYPE} revision=${REVISION:-latest}"
  python3 -c "import transformers, torch, lm_eval; print('transformers', transformers.__version__, '| torch', torch.__version__, '| lm_eval', getattr(lm_eval, '__version__', 'unknown'))"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
} 2>&1 | tee -a "${MANIFEST}"

FAILED=()
run () {  # run <tasks> <num_fewshot> <substitution seed> <run name>
  local extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  local margs="pretrained=${MODEL},parallelize=${PARALLELIZE},attn_implementation=sdpa,dtype=${DTYPE}"
  [[ -n "${REVISION}" ]] && margs+=",revision=${REVISION}"
  echo "=== $4 (shots=$2, CCE_SEED=$3) ==="
  if ! CCE_SEED="$3" ${LAUNCH} python3 -m lm_eval \
      --model hf \
      --model_args "${margs}" \
      --include_path "${TASK_DIR}" \
      --tasks "$1" \
      --device cuda --batch_size "${BATCH_SIZE}" \
      --num_fewshot "$2" \
      --seed "$3,$3,$3,${FEWSHOT_SEED}" \
      --output_path "${OUTDIR}/$4" --log_samples \
      "${extra[@]+"${extra[@]}"}"; then
    echo "!!! $4 FAILED"; FAILED+=("$4")
  fi
  echo "$4 finished $(date -Iseconds)" >> "${MANIFEST}"
}

for SHOTS in ${SHOTS_LIST}; do
  [[ "${RUN_ORIG}" == 1 ]] && run "$(tasks_for orig)" "${SHOTS}" 0 "orig_${SHOTS}shot"
  if [[ "${RUN_CCE}" == 1 ]]; then
    for s in ${SEEDS}; do
      run "$(tasks_for cce)" "${SHOTS}" "${s}" "cce_${SHOTS}shot_seed${s}"
    done
  fi
done

# Exhaustive design: 0-shot only; every removal is evaluated, so the seed has no effect.
[[ "${RUN_EXH}" == 1 ]] && run "$(tasks_for exh)" 0 0 "exh_0shot"

if (( ${#FAILED[@]} )); then echo "Finished with failures: ${FAILED[*]}"; exit 1; fi
echo "All runs finished. Results in ${OUTDIR}"
