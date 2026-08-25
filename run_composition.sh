#!/bin/bash -l
#SBATCH -J composition
#SBATCH -o /export/home/kaziz/motion/runs/composition_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 3:00:00
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
export COMP_BASE=${COMP_BASE:-latent}
export COMP_ITERS=${COMP_ITERS:-12}
export COMP_BETAS=${COMP_BETAS:-0,0.25,0.5,0.75,0.9,1.0}
export COMP_GND=${COMP_GND:-1}
export EVAL_N=${EVAL_N:-512}
$CPY constraint_composition.py
echo "=== COMPOSITION DONE ==="
