#!/bin/bash -l
#SBATCH -J guid_repl
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
# Replicated soft-guidance sweep. Now the load-bearing in-trajectory evidence,
# and currently single-seed. Set GR_L1=1 to add the unsquared-bone control arm.
set -e
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}"
export GR_REPS="${GR_REPS:-3}"
GR_BASE=latent $PY 04_guidance_replicated.py
GR_BASE=direct $PY 04_guidance_replicated.py
if [ "${GR_L1:-0}" = "1" ]; then
  GR_BASE=latent GR_L1=1 $PY 04_guidance_replicated.py
fi
echo "=== GUIDANCE REPLICATION DONE ==="
