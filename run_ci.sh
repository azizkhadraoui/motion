#!/bin/bash -l
#SBATCH -J replicated_ci
#SBATCH -o /export/home/kaziz/motion/runs/ci_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 6:00:00
# ---------------------------------------------------------------------------
# Replicated evaluation with 95% CIs (Baggag next-step #2).
# 4 key variants x CI_REPS(=20) replications, varied sampling seed, fixed clip set.
# ~4 variants x 20 reps x ~1-3 min each. Inference only on existing checkpoints.
#   sbatch run_ci.sh          # default 20 reps
#   CI_REPS=10 sbatch run_ci.sh   # fewer reps if time-constrained
# ---------------------------------------------------------------------------
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/envs/ml/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export MAIN_SCRIPT=$WORK/code/lfm_clfm_cdfm_experiment.py
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=
export CI_REPS=${CI_REPS:-20}
cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT  CI_REPS=$CI_REPS"
$CPY replicated_ci.py
echo "=== REPLICATED CI DONE ==="
