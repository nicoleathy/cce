#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ## 
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=120         # Number of CPU to request for the job
#SBATCH --mem=400GB                 # How much memory does your job require?
#SBATCH --gres=gpu:4                # Do you require GPUS? If not delete this line
#SBATCH --time=05-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=%u.%j.out          # Where should the log files go?
                                    # You must provide an absolute path eg /common/home/module/username/
                                    # If no paths are provided, the output file will be placed in your current working directory


################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=researchlong          # The partition you've been assigned
#SBATCH --account=tanahhweeresearch            # The account you've been assigned (normally student)
#SBATCH --qos=nicolet.2023-2024-11-21    # What is the QOS assigned to you? Check with myinfo command
#SBATCH --mail-user=                           # Who should receive the email notifications
#SBATCH --job-name=CCE  # Give the job a name

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Python/3.11.11-GCCcore-13.3.0
module load CUDA/11.8.0
 
# Use the virtual environment if it exists; otherwise fall back to the
# packages installed in ~/.local (what the previous jobs used).
VENV=~/myenv
if [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
else
  echo "Note: no virtual environment at ${VENV}; using ~/.local packages."
fi

if python3 -c "from huggingface_hub import whoami; whoami()" >/dev/null 2>&1; then
  echo "Hugging Face token OK."
else
  echo "No valid Hugging Face login found. Run 'hf auth login' on the login node first."
  exit 1
fi

# If you require any packages, install it as usual before the srun job submission.
# pip install matplotlib
# pip install seaborn
# pip install scikit-metrics
# pip install --upgrade transformers
# pip install --upgrade accelerate
# pip install lm-eval
# pip install torch


# srun --gres=gpu:4 python3 -m lm_eval \
#   --model hf \
#   --model_args pretrained=google/gemma-3-27b-pt,parallelize=True,attn_implementation=sdpa \
#   --tasks arc_challenge,commonsense_qa,hellaswag,piqa,winogrande \
#   --device cuda \
#   --batch_size auto \
#   --num_fewshot 0 \
#   --seed 0 \
#   --output_path /gscratch5/users/asalem/wicked/results.mcq/google/gemma-3-27b-pt/cce_0shot_seed0 \
#   --log_samples

# srun --gres=gpu:4 python3 -m lm_eval \
#   --model hf \
#   --model_args pretrained=google/gemma-3-27b-pt,parallelize=True,attn_implementation=sdpa \
#   --tasks arc_challenge,commonsense_qa,hellaswag,piqa,winogrande \
#   --device cuda \
#   --batch_size auto \
#   --num_fewshot 5 \
#   --seed 0 \
#   --output_path /gscratch5/users/asalem/wicked/results.mcq/google/gemma-3-27b-pt/cce_5shot_seed0 \
#   --log_samples

# srun --gres=gpu:4 python3 -m lm_eval \
#   --model hf \
#   --model_args pretrained=meta-llama/Llama-3.2-1B,parallelize=True,attn_implementation=sdpa \
#   --tasks arc_challenge,commonsense_qa,hellaswag,piqa,winogrande,winogrande_orig \
#   --device cuda \
#   --batch_size auto \
#   --num_fewshot 0 \
#   --seed 0 \
#   --output_path /gscratch5/users/asalem/wicked/results.mcq/meta-llama/Llama-3.2-1B/cce_0shot_seed0 \
#   --log_samples

# MODEL="mistralai/Mistral-7B-Instruct-v0.3"
# OUTDIR="/common/home/users/n/nicolet.2023/fallacies/results.mcq/${MODEL}"
# mkdir -p "${OUTDIR}"

# for s in 0 1 2 3 4; do
#   echo "=== seed ${s} ==="
#   CCE_SEED="${s}" srun --gres=gpu:4 python3 -m lm_eval \
#     --model hf \
#     --model_args "pretrained=${MODEL},parallelize=True,attn_implementation=sdpa" \
#     --tasks arc_challenge,commonsense_qa,hellaswag,piqa,winogrande,winogrande_orig \
#     --device cuda --batch_size auto --num_fewshot 0 --seed "${s}" \
#     --output_path "${OUTDIR}/cce_0shot_seed${s}" --log_samples \
#     || echo "!!! seed ${s} FAILED (exit $?)"
# done
 
set -uo pipefail
 
MODEL="${1:-}"
if [ -z "${MODEL}" ]; then
  echo "No model given. Usage: sbatch sbatchTemplatePython.sh <hf-model-id>"
  exit 1
fi
 
TASK_DIR=~/lm-evaluation-harness/lm_eval/tasks/cce_mcf     # adjust to where you unzipped cce_mcf
OUTDIR="/common/home/users/n/nicolet.2023/fallacies/results.mcf/${MODEL}"
mkdir -p "${OUTDIR}"
 
SHOTS_LIST="${SHOTS_LIST:-0 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FEWSHOT_SEED="${FEWSHOT_SEED:-1234}"
RUN_ORIG="${RUN_ORIG:-1}"
RUN_CCE="${RUN_CCE:-1}"
RUN_EXH="${RUN_EXH:-1}"
LIMIT="${LIMIT:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PARALLELIZE="${PARALLELIZE:-False}"
 
ORIG_TASKS=arc_challenge_mcf_orig,commonsense_qa_mcf_orig,hellaswag_mcf_orig,piqa_mcf_orig,winogrande_mcf_orig
CCE_TASKS=arc_challenge_mcf_cce,commonsense_qa_mcf_cce,hellaswag_mcf_cce,piqa_mcf_cce,winogrande_mcf_cce
EXH_TASKS=arc_challenge_mcf_exh,commonsense_qa_mcf_exh,hellaswag_mcf_exh,piqa_mcf_exh,winogrande_mcf_exh
 
# Substitution seed for the few-shot pool (train split). Fixed, so demonstrations
# are identical in every run; read by utils.py.
export CCE_FEWSHOT_SEED="${FEWSHOT_SEED}"
 
# Reduces memory fragmentation on long 5-shot prompts.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 
# Record the environment (for the Methodology section)
MANIFEST="${OUTDIR}/run_manifest.txt"
{
  echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) $(date -Iseconds) ==="
  echo "model=${MODEL}"
  echo "shots=${SHOTS_LIST} seeds=${SEEDS} fewshot_seed=${FEWSHOT_SEED} limit=${LIMIT:-none}"
  echo "batch_size=${BATCH_SIZE} parallelize=${PARALLELIZE} dtype=bfloat16"
  python3 --version 2>&1
  python3 -c "import transformers, torch, datasets; print('transformers', transformers.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| datasets', datasets.__version__)"
  python3 -c "import lm_eval; print('lm_eval', getattr(lm_eval, '__version__', 'unknown'))"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} 2>&1 | tee -a "${MANIFEST}"
 
# Preflight: stop now if the tokenizer can't load (missing packages, gated
# model, licence not accepted), instead of failing every run below.
if ! python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL}')"; then
  echo "Tokenizer for ${MODEL} failed to load; fix the environment before running."
  exit 1
fi
echo "Tokenizer for ${MODEL} OK."
 
FAILED=()
 
run () {  # run <tasks> <num_fewshot> <eval_seed> <output-subfolder>
  local extra=()
  [[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
  echo "=== $4 (shots=$2, CCE_SEED=$3, fewshot_seed=${FEWSHOT_SEED}, batch=${BATCH_SIZE}) ==="
  if ! CCE_SEED="$3" srun --gres=gpu:4 python3 -m lm_eval \
      --model hf \
      --model_args "pretrained=${MODEL},parallelize=${PARALLELIZE},attn_implementation=sdpa,dtype=bfloat16" \
      --include_path "${TASK_DIR}" \
      --tasks "$1" \
      --device cuda --batch_size "${BATCH_SIZE}" \
      --num_fewshot "$2" \
      --seed "$3,$3,$3,${FEWSHOT_SEED}" \
      --output_path "${OUTDIR}/$4" --log_samples \
      "${extra[@]+"${extra[@]}"}"; then
    echo "!!! $4 FAILED"
    FAILED+=("$4")
  fi
  echo "$4 finished $(date -Iseconds)" >> "${MANIFEST}"
}
 
for SHOTS in ${SHOTS_LIST}; do
  if [[ "${RUN_ORIG}" == 1 ]]; then
    run "${ORIG_TASKS}" "${SHOTS}" 0 "orig_${SHOTS}shot"          # deterministic: run once
  fi
  if [[ "${RUN_CCE}" == 1 ]]; then
    for s in ${SEEDS}; do
      run "${CCE_TASKS}" "${SHOTS}" "${s}" "cce_${SHOTS}shot_seed${s}"
    done
  fi
done
 
# Exhaustive design: 0-shot only; every removed option is evaluated, so the seed has no effect.
if [[ "${RUN_EXH}" == 1 ]]; then
  run "${EXH_TASKS}" 0 0 "exh_0shot"
fi
 
if (( ${#FAILED[@]} )); then
  echo "Finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "All runs finished. Results in ${OUTDIR}"




