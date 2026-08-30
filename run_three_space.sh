#!/bin/bash -l
#SBATCH -J threespace
#SBATCH -o /export/home/kaziz/motion/runs/threespace_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 4:00:00
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
export TS_REPS=${TS_REPS:-5}
export TS_NFE=${TS_NFE:-1,2,4,8,16}
export TS_BASES=${TS_BASES:-latent,direct}
export EVAL_N=${EVAL_N:-512}
$CPY three_space_geometry.py
echo "=== THREE SPACE DONE ==="
