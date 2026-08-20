#!/bin/bash -l
#SBATCH -J todo4
#SBATCH -o /export/home/kaziz/motion/runs/todo4_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 5:00:00
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/envs/ml/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export MAIN_SCRIPT=$WORK/code/lfm_clfm_cdfm_experiment.py
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=
cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT"
export T4_BATCH=${T4_BATCH:-32}
export T4_WEIGHTS=${T4_WEIGHTS:-0.25,1.0,2.0}
export T4_EVAL=${T4_EVAL:-1}
$CPY todo4_guidance_sanity.py
echo "=== TODO4 GUIDANCE SANITY DONE ==="
