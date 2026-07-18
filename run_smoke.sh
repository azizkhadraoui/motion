#!/bin/bash -l
#SBATCH -J clfm_smoke
#SBATCH -o /export/home/kaziz/motion/runs/smoke_%j.out
#SBATCH -p gpu-dev
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 0:30:00
# ---------------------------------------------------------------------------
# SMOKE TEST: 200 steps, latent base only, full eval+projection+W&B.
# Run this FIRST. Confirm a W&B run appears at wandb.ai/kaziz/motion-clfm with
# a loss curve, an eval FID, and (since VARIANT=all here) the projection table.
# Only after this succeeds should you launch the full 300k jobs.
# ---------------------------------------------------------------------------
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=kaziz
export WANDB_RUN=smoke
# export WANDB_API_KEY=...   # set this (from wandb.ai/authorize) OR run `wandb login` once
export SMOKE_TEST=1
export VARIANT=all           # smoke runs the whole tiny pipeline incl. the table
export USE_WANDB=1

echo "RVQ_CKPT=$RVQ_CKPT"
mkdir -p $WORK_DIR
cd $WORK/code
$CPY lfm_clfm_cdfm_experiment.py
echo "=== SMOKE DONE — check wandb.ai/kaziz/motion-clfm ==="