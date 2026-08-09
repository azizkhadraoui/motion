#!/bin/bash -l
#SBATCH -J clfm_ablation
#SBATCH -o /export/home/kaziz/motion/runs/ablation_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
# ---------------------------------------------------------------------------
# CLFM in-process schedule ablation. Loads the trained latent checkpoint and sweeps
# the projection schedule (window: final 10/20/50% of ODE steps; stride: every 1/2/4 steps),
# plus unconstrained and post-hoc reference points. ~11 configs x ~1024 clips.
# Determines whether the FID-55.9 in-process collapse is structural or a tuning artifact.
#   sbatch run_ablation.sh
# ---------------------------------------------------------------------------
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/envs/ml/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export MAIN_SCRIPT=$WORK/code/lfm_clfm_cdfm_experiment.py
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=
export WANDB_RUN=clfm_inproc_ablation
# export WANDB_API_KEY=...   # set on cluster, or rely on ~/.netrc

cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT"
$CPY ablation_inproc.py
echo "=== ABLATION DONE — table + curves at wandb.ai/<entity>/motion-clfm ==="
