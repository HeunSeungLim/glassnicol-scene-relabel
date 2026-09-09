#!/usr/bin/env bash
# Pull the two external-baseline arms from the H200 host and score every cell on the official test splits.
#
# The comparison the paper was missing is against methods that already exist for label noise, rather than
# only against our own earlier vote.  Two are trained here on the same roster and budget as every other arm:
#   lsmooth  training on the released labels with classification label smoothing, i.e. absorbing the noise
#            instead of repairing it;
#   confid   replacing a type id when a detector that never saw the box disagrees confidently, the
#            confident-learning family.
# Scoring runs on the CPU here for every cell, the same environment as the existing arms, so the numbers
# sit in one table without a device caveat.
set -Eeuo pipefail
P=${WORKDIR}
SV=$P/seed_v1; SC=$P/scenes_v1
# Set REMOTE to the host that ran the accelerator-A cells, or leave it empty to score local runs only.
REMOTE=${REMOTE:-}
R=${REMOTE_RUNS:-/path/to/runs}
SCP="scp -q -o ConnectTimeout=15"

A_RUNS="scene_lsmooth_s1337 scene_lsmooth_s3407 scene_lsmooth_s4567 scene_confid_s1337 scene_confid_s3407 scene_confid_s4567"
B_RUNS="scene136_lsmooth_s1337 scene136_lsmooth_s5678 scene136_confid_s1337 scene136_confid_s5678"

for n in $A_RUNS; do
  if [ -z "$REMOTE" ] || ! ssh -o ConnectTimeout=15 "$REMOTE" "test -f $R/$n/FINAL_RESULT.json" 2>/dev/null; then
    echo "NOT READY $n"; continue
  fi
  mkdir -p "$SV/runs/$n/train/weights"
  for jf in FINAL_RESULT.json CONTRACT.json; do
    [ -f "$SV/runs/$n/$jf" ] || $SCP "$REMOTE:$R/$n/$jf" "$SV/runs/$n/"
  done
  [ -f "$SV/runs/$n/train/weights/best.pt" ] || $SCP "$REMOTE:$R/$n/train/weights/best.pt" "$SV/runs/$n/train/weights/"
  echo "PULLED $n"
done

cd "$SC"
for n in $A_RUNS $B_RUNS; do
  d=$SV/runs/$n
  if [ ! -f "$d/FINAL_RESULT.json" ] || [ ! -f "$d/train/weights/best.pt" ]; then echo "MISSING $n"; continue; fi
  if [ -f "evals_official/ood_$n/RESULT.json" ] && [ -f "evals_official/standard_$n/RESULT.json" ]; then
    echo "SCORED $n"; continue
  fi
  python3 eval_official_v1.py "$d" "$n" --device cpu | grep -E 'standard|ood' || echo "EVAL FAILED $n"
done
echo PREFETCH_BASELINES_DONE
