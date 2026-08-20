#!/bin/bash -l
#SBATCH -J ldf
#SBATCH -o /export/home/kaziz/motion/runs/ldf_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 16:00:00
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
export LDF_C=${LDF_C:-0.1,1.0,10.0}
export LDF_P=${LDF_P:-1,2,3}
export LDF_STEPS=${LDF_STEPS:-50,100}
export LDF_BASES=${LDF_BASES:-direct,latent}
export EVAL_N=${EVAL_N:-512}
$CPY ldf_dual_flow.py
echo "=== LDF DONE ==="
