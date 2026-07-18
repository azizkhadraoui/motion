#!/bin/bash -l
#SBATCH -J clfm_eval
#SBATCH -o /export/home/kaziz/motion/runs/eval_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH --time 1-00:00:00
# ---------------------------------------------------------------------------
# Run AFTER all four training jobs finish. Loads every trained base, runs all
# projection variants (inference-only), emits the full comparison table, the
# qualitative figures, and the GIF comparison — all logged to W&B.
#   sbatch run_eval.sh
# ---------------------------------------------------------------------------
set -e
export WORK=/export/home/kaziz/motion
export CPY=$WORK/miniconda3/bin/python
export HML3D_ROOT=/export/home/kaziz/motion/data/humanml3d_extracted/HumanML3D/humanml
export RVQ_CKPT=$(find $WORK -name rvq_vae_best.pt 2>/dev/null | head -1)
export WORK_DIR=$WORK/runs
export WANDB_PROJECT=motion-clfm
export WANDB_ENTITY=kaziz
export WANDB_RUN=eval_table
# export WANDB_API_KEY=...
export SMOKE_TEST=0
export VARIANT=eval          # skips training, runs the table + figures + GIFs
export EVAL_N=1024
export USE_WANDB=1

cd $WORK/code
$CPY lfm_clfm_cdfm_experiment.py
echo "=== EVAL DONE — table + figures + GIFs logged to wandb.ai/kaziz/motion-clfm ==="