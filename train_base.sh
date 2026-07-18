#!/bin/bash -l
#SBATCH -J clfm_train
#SBATCH -o /export/home/kaziz/motion/runs/train_%x_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 3-00:00:00
# ---------------------------------------------------------------------------
# Trains ONE base to 300k steps. Submit four times, one per VARIANT, so they run
# in parallel on four V100s (finishes the matrix in ~1 training's wall-clock):
#
#   sbatch --job-name=latent      --export=ALL,VARIANT=latent      train_base.sh
#   sbatch --job-name=latent_pen  --export=ALL,VARIANT=latent_pen  train_base.sh
#   sbatch --job-name=direct      --export=ALL,VARIANT=direct      train_base.sh
#   sbatch --job-name=direct_pen  --export=ALL,VARIANT=direct_pen  train_base.sh
#
# Resumable: if a job is killed, resubmit the same line — it continues from the
# last checkpoint (saved every 2000 steps in $WORK_DIR).
# ---------------------------------------------------------------------------
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(ls $WORK/**/rvq_vae_best.pt $WORK/data/**/rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=kaziz
export WANDB_RUN=${VARIANT}
# export WANDB_API_KEY=...   # set once (env or `wandb login`)
export SMOKE_TEST=0
export FULL_STEPS=300000
export USE_WANDB=1
# VARIANT is passed via --export on the sbatch line (latent|latent_pen|direct|direct_pen)

echo "VARIANT=$VARIANT  RVQ_CKPT=$RVQ_CKPT  steps=$FULL_STEPS"
mkdir -p $WORK_DIR
cd $WORK/code
$CPY lfm_clfm_cdfm_experiment.py
echo "=== [$VARIANT] training done ==="
