#!/bin/bash
# Example SLURM wrapper. Fill in the partition/account lines for your cluster,
# and create the log folder first: mkdir -p logs
#
#   sbatch scripts/slurm_example.sbatch scripts/run_cce.sh meta-llama/Llama-3.2-3B outputs/main
#   sbatch scripts/slurm_example.sbatch scripts/run_robustness.sh chat
#
#SBATCH --job-name=cce
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x.%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -uo pipefail
mkdir -p logs
# Activate your environment here, e.g.:  source .venv/bin/activate
export LAUNCH="srun --gres=gpu:4"
bash "$@"
