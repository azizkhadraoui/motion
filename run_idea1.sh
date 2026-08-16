#!/bin/bash -l
#SBATCH -J idea1_guidance
#SBATCH -o /export/home/kaziz/motion/runs/idea1_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
# ---------------------------------------------------------------------------
# IDEA 1: soft constraint-guidance sweep (latent vs direct). Shows in-latent guidance
# fails structurally like hard projection, while direct-space guidance works.
# 2 bases x 7 guidance weights x ~1024 clips. Inference only on existing checkpoints.
#   sbatch run_idea1.sh
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
export WANDB_RUN=idea1_soft_guidance
# export WANDB_API_KEY=...   # set on cluster, or rely on ~/.netrc

cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT"
$CPY idea1_soft_guidance.py
echo "=== IDEA 1 DONE — table + curves at wandb.ai/<entity>/motion-clfm ==="
