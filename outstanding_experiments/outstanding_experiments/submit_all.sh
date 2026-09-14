#!/bin/bash
# submit_all.sh -- submit the outstanding GPU experiments.
#
# All four are inference-only on frozen checkpoints and share no state, so they are
# independent as JOBS and can run concurrently. They are not independent as EVIDENCE:
# job 01 decides how the paper is framed, so if GPU budget is tight run it alone first
# and decide before spending the rest.
#
#   bash submit_all.sh gate      01 only. The default, and the honest one.
#   bash submit_all.sh parallel  all four at once, if the queue allows it
#   bash submit_all.sh after     02-04 only, once 01 has landed
#   bash submit_all.sh chain     all four, each starting when the previous finishes
#
# Non-GPU work that should happen WHILE these run, on the login node:
#   bash 06_lookups.sh > lookups.txt        compute figures, clip counts, best-of-N
#   python 05_paired_ci.py <per-rep file>   the R@3 and FSR paired intervals
#
set -eu
MODE="${1:-gate}"
cd "$(dirname "$0")"

for f in 01_inproc_restd.py 02_attainability_latdisp.py 03_penalty_balance.py \
         04_guidance_replicated.py lfm_clfm_cdfm_experiment.py; do
  [ -f "$f" ] || { echo "missing: $f"; echo "copy this whole directory into the code"; \
                   echo "checkout so the scripts sit beside the main experiment file."; exit 1; }
done
[ -f env.sh ] || echo "note: no env.sh here; the launchers will look for ENV_SH"

sub () { echo -n "  $1  -> "; sbatch ${2:-} "$1" | awk '{print $NF}'; }

case "$MODE" in
  gate)
    echo "Submitting the decision gate only."
    sub 01_run.sh
    echo
    echo "When it lands, check three things in order:"
    echo "  1. the two reference rows reproduce ~0.147 unconstrained and ~0.142 post-hoc"
    echo "     -- if not, the harness is wrong and nothing below it means anything"
    echo "  2. BLE_pre vs BLE_post on the +final proj rows: if BLE_post is 0 and BLE_pre"
    echo "     is not, the published BLE column came from the final projection"
    echo "  3. restd ON vs off at matched k -- this is the result that decides the framing"
    echo
    echo "Then: bash submit_all.sh after"
    ;;
  parallel)
    echo "Submitting all four concurrently."
    for f in 01_run.sh 02_run.sh 03_run.sh 04_run.sh; do sub "$f"; done
    echo
    echo "Four V100s for roughly 45 min, 2.5 h, 30 min and 3.5 h respectively."
    ;;
  after)
    echo "Submitting 02-04 concurrently."
    for f in 02_run.sh 03_run.sh 04_run.sh; do sub "$f"; done
    ;;
  chain)
    echo "Submitting all four as a dependency chain (one GPU at a time)."
    prev=""
    for f in 01_run.sh 02_run.sh 03_run.sh 04_run.sh; do
      if [ -z "$prev" ]; then id=$(sbatch "$f" | awk '{print $NF}')
      else id=$(sbatch --dependency=afterok:"$prev" "$f" | awk '{print $NF}'); fi
      echo "  $f  -> $id${prev:+  (after $prev)}"
      prev="$id"
    done
    ;;
  *) echo "usage: bash submit_all.sh [gate|parallel|after|chain]"; exit 1 ;;
esac

echo
echo "watch:   squeue -u $USER"
echo "outputs: \$WORK_DIR/{inproc_restd_*,todo11_attainability_latdisp,penalty_balance_*,guidance_replicated_*}.json"
