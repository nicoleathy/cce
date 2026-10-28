#!/bin/bash
# Run from the repo root (the folder containing lm-evaluation-harness/ and scripts/).
# Create the log folder once first:  mkdir -p logs
#
# Submit one model:   sbatch scripts/submit_cce.sbatch <hf-model-id> <output-root>
# Submit several:     for m in meta-llama/Llama-3.2-3B mistralai/Mistral-7B-v0.3; do
#                       sbatch scripts/submit_cce.sbatch "$m" "$HOME/cce_results"; done
#
# Copy your cluster's partition / account / QOS lines from your old template
# into the block below. Do NOT copy the HF token line: authenticate once with
# `huggingface-cli login`, or export HF_TOKEN in ~/.bashrc.

#SBATCH --job-name=cce_mcf
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}/lm-evaluation-harness"
bash ../scripts/run_cce_mcf.sh "$1" "$2"
