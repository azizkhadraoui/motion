#!/bin/bash -l
#SBATCH -J attain_latdisp
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 3:00:00
# Attainability with latent displacement + a wider lambda sweep.
# One job settles four things: the displacement question, the gap in the locality
# curve, the degenerate P1 control, and the hardcoded round-trip drift.
set -e
source "${ENV_SH:-$(dirname "${BASH_SOURCE[0]}")/env.sh}"
export AT_LAMBDAS="${AT_LAMBDAS:-0.0003,0.001,0.003,0.01,0.03,0.1,1.0,10.0}"
$PY 02_attainability_latdisp.py
echo "=== ATTAINABILITY DONE ==="
