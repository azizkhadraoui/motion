#!/bin/bash -l
#SBATCH -J iclr_smoke
#SBATCH -o /export/home/kaziz/motion/runs/iclr_smoke_%j.out
#SBATCH -p gpu-all
#SBATCH --gres gpu:v100_16GB:1
#SBATCH -c 8
#SBATCH --mem 48000MB
#SBATCH --time 2:00:00
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
# tiny end-to-end pass over every new ICLR script; catches bugs before the long jobs
export USE_WANDB=0
set -x
EVAL_N=64 MF_STEPS=5 $CPY manifold_projection.py
LC_N=64 LC_REPS=1 LC_W=0,20 $CPY linear_constraint_placement.py
MR_N=64 MR_REPS=2 MR_BON=2 $CPY metric_robustness.py
OS_N=64 OS_REPS=1 OS_NFE=1,2 $CPY onestep_placement.py
IR_N=64 IR_REPS=2 $CPY inproc_direct_replicated.py
MF_SUFFIX=_smoke MF_STEPS=20 MF_EVAL_EVERY=10 $CPY meanflow_train.py --variant mf
MF_SUFFIX=_smoke MF_STEPS=20 MF_EVAL_EVERY=10 $CPY meanflow_train.py --variant imf
rm -f $WORK_DIR/clfm/ckpt/meanflow_*_smoke_*.pt
echo "=== SMOKE DONE ==="
