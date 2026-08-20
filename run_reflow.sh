#!/bin/bash -l
#SBATCH -J reflow
#SBATCH -o /export/home/kaziz/motion/runs/reflow_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 20:00:00
# ---------------------------------------------------------------------------
# Reflow / trajectory rectification, overnight. Three phases in one job:
#   1. generate REFLOW_PAIRS teacher pairs   (~1 h at 20k)
#   2. train the student on them             (~7 h at 40k steps)
#   3. NFE sweep assessment                  (~1.5 h)
# Resumable: pair shards are cached under $WORK_DIR/reflow_pairs and the student
# checkpoints every 2000 steps, so resubmitting the same line continues.
# Writes ONLY reflow_*.pt -- never touches latent_*.pt.
#   sbatch run_reflow.sh
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
export WANDB_RUN=reflow_rectification
export REFLOW_PAIRS=${REFLOW_PAIRS:-20000}
export REFLOW_STEPS=${REFLOW_STEPS:-40000}
export REFLOW_LR=${REFLOW_LR:-5e-5}
export REFLOW_NFE=${REFLOW_NFE:-1,2,4,8,16,50}
export EVAL_N=${EVAL_N:-1024}
cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT  pairs=$REFLOW_PAIRS  steps=$REFLOW_STEPS  EVAL_N=$EVAL_N"
$CPY reflow_experiment.py
echo "=== REFLOW DONE ==="
