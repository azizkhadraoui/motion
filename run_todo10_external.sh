#!/bin/bash -l
#SBATCH -J todo10_external
#SBATCH -o /export/home/kaziz/motion/runs/todo10_external_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 1:00:00
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
export EXT_DIR=${EXT_DIR:-}
export EXT_FORMAT=${EXT_FORMAT:-raw263}
export EXT_NAME=${EXT_NAME:-external}
export EXT_LIMIT=${EXT_LIMIT:-512}
$CPY todo10_external_eval.py
echo "=== TODO10_EXTERNAL DONE ==="
