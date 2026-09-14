#!/bin/bash
# 06_lookups.sh -- values that are outstanding but are NOT experiments.
# Run on the login node. No GPU, no scheduler, a few seconds.
#
#   bash 06_lookups.sh > lookups.txt
#
set -u
CODE="${CODE_DIR:-$HOME/motion/code}"
RUNS="${WORK_DIR:-$HOME/motion/runs}"
cd "$CODE" 2>/dev/null || { echo "set CODE_DIR"; exit 1; }

echo "==================== 1. COMPUTE AND WALL-CLOCK ===================="
echo "Checklist question 8 cannot be answered without this; asked for twice."
sacct -u "$USER" -S 2026-07-01 --format=JobName%22,Elapsed,AllocTRES%42,State 2>/dev/null \
  | head -80
echo
echo "  total GPU time, completed jobs only:"
sacct -u "$USER" -S 2026-07-01 -X --state=COMPLETED --format=Elapsed -n 2>/dev/null \
  | awk -F: '{s+=$1*3600+$2*60+$3} END {printf "    %.1f hours across %d jobs\n", s/3600, NR}'

echo
echo "==================== 2. CLIP COUNT PER TABLE ===================="
echo "The report asserts 1,024 and flags it as ambiguous with 512. FID is strongly"
echo "sample-size dependent, so every caption needs the number."
grep -rn "EVAL_N\|N=512\|N=1024\|T11_N\|AT_N" *.py 2>/dev/null | head -20
echo "  --- as actually used, from the run logs ---"
grep -rhoE "[0-9]+ clips" wandb/run-*/files/output.log 2>/dev/null | sort | uniq -c | sort -rn | head

echo
echo "==================== 3. BEST-OF-N VALUES ===================="
echo "Marked as needing data in Chapter 6. A log lookup, not a run."
grep -rhE "best.of.[0-9]|N=[0-9]+ .*FID|bestofn" wandb/run-*/files/output.log 2>/dev/null | head -20

echo
echo "==================== 4. SAMPLER SWEEP, REMAINING ROWS ===================="
echo "Three of eight are named in the draft; the rest are appendix completeness."
grep -rh -A 14 "INFERENCE-TIME FID ABLATION\|sampler ablation" \
  wandb/run-*/files/output.log 2>/dev/null | head -24

echo
echo "==================== 5. PER-REPLICATION VALUES ===================="
echo "05_paired_ci.py needs these; replicated_ci.json was not in the archive sent."
ls -l "$RUNS"/replicated_ci*.json 2>/dev/null || echo "  NOT FOUND in $RUNS"
grep -l "REPLICATED" wandb/run-*/files/output.log 2>/dev/null | head -3
