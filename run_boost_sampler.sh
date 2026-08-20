#!/bin/bash -l
#SBATCH -J boost_sampler
#SBATCH -o /export/home/kaziz/motion/runs/boost_sampler_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 6:00:00
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
export BOOST_BASE=${BOOST_BASE:-latent}
export BOOST_GI=${BOOST_GI:-0.0-1.0,0.1-1.0,0.2-1.0,0.1-0.9,0.2-0.8,0.3-0.9,0.0-0.8}
export BOOST_CHURN=${BOOST_CHURN:-0,0.25,0.5,1.0,2.0}
export EVAL_N=${EVAL_N:-1024}
$CPY boost_sampler.py
echo "=== BOOST_SAMPLER DONE ==="
