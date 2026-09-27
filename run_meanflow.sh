#!/bin/bash -l
#SBATCH -J meanflow
#SBATCH -o /export/home/kaziz/motion/runs/meanflow_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 14:00:00
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/envs/ml/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export MAIN_SCRIPT=$WORK/code/lfm_clfm_cdfm_experiment.py
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=
export PYTHONUNBUFFERED=1
cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT"
# usage: sbatch run_meanflow.sh mf|imf
VAR=${1:-${MF_VARIANT:-mf}}
echo "variant=$VAR"
$CPY meanflow_train.py --variant $VAR
echo "=== MEANFLOW $VAR DONE ==="
