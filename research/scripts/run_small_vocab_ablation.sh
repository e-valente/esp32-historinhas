#!/usr/bin/env bash
# Historinhas small-vocabulary control at vocab 4096: all five arms, both
# seeds, core-matched at 1.5M. Its numbers are a new pt_BR cohort, not a reuse
# of the deployment vocabulary table.
#
# Fails immediately if any run fails: a partial ablation is not a result.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

for seed in 0 1; do
  for arm in baseline ple ple_notable fatembed bigcore; do
    log "RUN $arm seed$seed"
    uv run python -m research.train \
      --arm "$arm" --vocab 4096 --target-core 1500000 --steps 3000 \
      --seed "$seed" --tag clean
  done
done

log "small-vocab ablation complete"
