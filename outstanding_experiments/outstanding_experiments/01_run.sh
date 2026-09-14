#!/bin/bash -l
#SBATCH -J inproc_restd
#SBATCH -o slurm-%x-%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 2:00:00
# ---------------------------------------------------------------------------
# Resolves the two open questions on the in-ODE projection path.
#
#   Q1  Where is BLE measured? sample() applies a FINAL projection to mode="inproc"
#       as well as to mode="posthoc", so the BLE=0.0 column of the published
#       in-process ablation may be produced by that final step rather than by the
#       in-trajectory ones. This reports BLE both before and after it.
#
#   Q2  Is the collapse the missing re-standardization? The in-ODE branch decodes
#       with z*z_std+z_mean and re-encodes without inverting it. This runs both
#       arms side by side at matched stride.
#
# Produces inproc_restd_latent.json (and _direct.json) in $WORK_DIR. Inference only,
# no training, no checkpoint is modified. The main experiment script is NOT patched:
# the fix is applied inside a local copy of the sampler, so nothing else changes.
#
#   sbatch run_inproc_restd.sh
#
# The #SBATCH lines are site-specific; adjust partition and GPU type as needed.
# ---------------------------------------------------------------------------
set -e
ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/slurm/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -f "$ENV_SH" ] || ENV_SH="${SLURM_SUBMIT_DIR:-$PWD}/env.sh"
source "$ENV_SH"

export ID_N=${ID_N:-512}
export ID_WINDOW=${ID_WINDOW:-0.10}     # 10% window, the best cell of the published ablation
export ID_STRIDES=${ID_STRIDES:-4,2,1}  # k=4 gave FID 21.2, k=1 gave 55.6

# Latent is the arm under test: the missing re-standardization exists only there.
export ID_BASE=latent
$PY 01_inproc_restd.py

# Direct is the control. It has no encoder in the loop, so both arms are identical
# by construction and it should reproduce the published 0.158 either way. If it does
# not, something else changed since August and the comparison is not clean.
export ID_BASE=direct
export ID_STRIDES=4
$PY 01_inproc_restd.py

echo "=== INPROC RESTD DONE ==="
