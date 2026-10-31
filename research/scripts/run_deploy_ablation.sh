#!/usr/bin/env bash
# Deployment-oriented Historinhas ablation at vocab 32768: three arms, both
# seeds, core-matched at ~559k so every arm targets the ESP32 memory budget.
#
# Produces a new pt_BR cohort and checkpoints that can be exported after review.
#
# bs16/sl256 keeps 32k-class cross-entropy off the MPS memory cliff; at
# bs32/sl512 it stalls.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for seed in 0 1; do
  for arm in baseline ple fatembed; do
    log "RUN $arm seed$seed"
    uv run python -m research.train \
      --arm "$arm" --vocab 32768 --d-model 96 --n-layers 6 --ple-dim 128 \
      --target-core 560000 --batch-size 16 --seq-len 256 --steps 5000 \
      --seed "$seed" --tag cleandeploy
  done
done

log "deploy ablation complete"
