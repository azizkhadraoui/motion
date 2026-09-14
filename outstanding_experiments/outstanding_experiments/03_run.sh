#!/bin/bash -l
#SBATCH -J penalty_balance
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 0:30:00
# Does the bone term actually steer the guidance? Both pipelines.
set -e
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}"
PB_BASE=latent $PY 03_penalty_balance.py
PB_BASE=direct $PY 03_penalty_balance.py
echo "=== PENALTY BALANCE DONE ==="
