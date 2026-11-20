#!/bin/bash
# Example Slurm wrapper. Add your cluster's partition/account lines.
# Submit from the repository root:
#   mkdir -p logs && sbatch scripts/slurm_example.sbatch meta-llama/Llama-3.2-3B results/runs <revision>
#SBATCH --job-name=cce
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
LAUNCH="srun --gres=gpu:4" bash scripts/run_cce.sh "$@"
