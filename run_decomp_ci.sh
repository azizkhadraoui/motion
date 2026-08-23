#!/bin/bash -l
#SBATCH -J decomp_ci
#SBATCH -o /export/home/kaziz/motion/runs/decomp_ci_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 20:00:00
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
export DECOMP_BASES=${DECOMP_BASES:-latent,direct}
export CI_REPS=${CI_REPS:-20}
export EVAL_N=${EVAL_N:-1024}
$CPY decomp_ci.py
echo "=== DECOMP_CI DONE ==="
