#!/bin/bash
# Run the multiple-choice-format CCE tasks (lm_eval/tasks/cce_mcf) for one model.
#
# Usage (from inside lm-evaluation-harness/):
#   bash ../scripts/run_cce_mcf.sh <hf-model-id> <output-root>
#
# Example:
#   bash ../scripts/run_cce_mcf.sh meta-llama/Llama-3.2-3B "$HOME/cce_results"
#
# What it runs:
#   orig_{S}shot          Orig tasks, once per shot setting (deterministic)
#   cce_{S}shot_seed{N}   random-substitution CCE, eval seeds N = 0..4
#   exh_0shot             exhaustive CCE (every item x every removed option)
#
# Optional environment variables:
#   SHOTS_LIST="0 5"     shot settings for orig/cce (5-shot is plain few-shot,
#                        letter scoring, NOT chain-of-thought)
#   SEEDS="0 1 2 3 4"    CCE eval seeds
#   FEWSHOT_SEED=1234    fixed seed for (a) which demonstrations lm-eval samples
#                        and (b) the substitution applied to them, so every run
#                        sees the same demonstrations
#   RUN_ORIG=1 RUN_CCE=1 RUN_EXH=1   set any to 0 to skip that family
#   LAUNCH="srun --gres=gpu:4"       set LAUNCH="" to run without srun
#   LIMIT=20             pass --limit for a quick smoke test
#
# Hugging Face auth: run `huggingface-cli login` once on the cluster, or export
# HF_TOKEN in your shell profile. Never write the token into this script.

set -uo pipefail

MODEL="${1:?usage: run_cce_mcf.sh <hf-model-id> <output-root>}"
OUTROOT="${2:?usage: run_cce_mcf.sh <hf-model-id> <output-root>}"
OUTDIR="${OUTROOT}/${MODEL}"

SHOTS_LIST="${SHOTS_LIST:-0 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FEWSHOT_SEED="${FEWSHOT_SEED:-1234}"
RUN_ORIG="${RUN_ORIG:-1}"
RUN_CCE="${RUN_CCE:-1}"
RUN_EXH="${RUN_EXH:-1}"
LAUNCH="${LAUNCH-srun --gres=gpu:4}"
LIMIT="${LIMIT:-}"

DATASETS="arc_challenge commonsense_qa hellaswag piqa winogrande"
tasks_for () {  # tasks_for <orig|cce|exh>
  local out=""
  for d in ${DATASETS}; do out="${out:+${out},}${d}_mcf_$1"; done
  echo "${out}"
}

export CCE_FEWSHOT_SEED="${FEWSHOT_SEED}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/run_manifest.txt"
{
  echo "model=${MODEL}"
  echo "date=$(date -Iseconds)"
  echo "fewshot_seed=${FEWSHOT_SEED} shots=${SHOTS_LIST} seeds=${SEEDS} limit=${LIMIT:-none}"
  echo "harness_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} >> "${LOG}"

FAILED=()

run () {  # run <tasks> <num_fewshot> <eval_seed> <outname>
  local extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  echo "=== $4 (tasks=$1, shots=$2, CCE_SEED=$3, fewshot_seed=${FEWSHOT_SEED}) ==="
  if ! CCE_SEED="$3" ${LAUNCH} python3 -m lm_eval \
      --model hf \
      --model_args "pretrained=${MODEL},parallelize=True,attn_implementation=sdpa" \
      --tasks "$1" \
      --device cuda --batch_size auto \
      --num_fewshot "$2" \
      --seed "$3,$3,$3,${FEWSHOT_SEED}" \
      --output_path "${OUTDIR}/$4" \
      --log_samples \
      "${extra[@]+"${extra[@]}"}"; then
    echo "!!! $4 FAILED"
    FAILED+=("$4")
  fi
  echo "$4 done $(date -Iseconds)" >> "${LOG}"
}

for SHOTS in ${SHOTS_LIST}; do
  if [[ "${RUN_ORIG}" == 1 ]]; then
    run "$(tasks_for orig)" "${SHOTS}" 0 "orig_${SHOTS}shot"
  fi
  if [[ "${RUN_CCE}" == 1 ]]; then
    for s in ${SEEDS}; do
      run "$(tasks_for cce)" "${SHOTS}" "${s}" "cce_${SHOTS}shot_seed${s}"
    done
  fi
done

# Exhaustive design: 0-shot only. The eval seed has no effect (every j is used).
if [[ "${RUN_EXH}" == 1 ]]; then
  run "$(tasks_for exh)" 0 0 "exh_0shot"
fi

if (( ${#FAILED[@]} )); then
  echo "Finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "All runs finished. Results in ${OUTDIR}"
