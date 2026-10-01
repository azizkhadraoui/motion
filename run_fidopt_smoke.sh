#!/bin/bash -l
#SBATCH -J fo_smoke
#SBATCH -o /export/home/kaziz/motion/runs/fidopt_smoke_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
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
export PYTHONUNBUFFERED=1
cd $WORK/code
echo "RVQ_CKPT=$RVQ_CKPT"
# tiny end-to-end pass over the FID-optimisation scripts (latent base; separate checkpoint tag)
set -x
export FO_TAG=_smoke FO_N=32 FO_REPS=2 FO_SAVE_AT=10,20 FO_EVAL_STEPS=10,20 NEON_NS=128
for t in neon auto dfm ftctl; do $CPY fidopt_train.py --task $t --base latent; done
for e in neon auto dfm; do $CPY fidopt_eval.py --exp $e --base latent; done
FO_TAG=_smoke $CPY fidopt_eval.py --exp guidance --base latent
$CPY fidopt_train.py --task ftctl --base direct
rm -f $WORK_DIR/clfm/ckpt/fidopt_*_smoke_*.pt
echo "=== FO SMOKE DONE ==="
