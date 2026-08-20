#!/bin/bash -l
#SBATCH -J boost_bestofn
#SBATCH -o /export/home/kaziz/motion/runs/boost_bestofn_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 8:00:00
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
export BOOST_N=${BOOST_N:-1,2,4,8}
export BOOST_PROXY=${BOOST_PROXY:-ble,fsr,random}
export EVAL_N=${EVAL_N:-1024}
$CPY boost_bestofn.py
echo "=== BOOST_BESTOFN DONE ==="
