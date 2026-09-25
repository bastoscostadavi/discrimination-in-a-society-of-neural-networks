#!/bin/bash
# Validate and merge completed sweep shards after the compute array succeeds.
# WORKERS must match the compute jobs because it is part of the cache key.
#SBATCH --job-name=ednna-merge
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.out

set -euo pipefail

PRESET=${PRESET:-full}
N_AGENTS=${N_AGENTS:-200}
BATCH_SIZE=${BATCH_SIZE:-256}
WORKERS=${WORKERS:-48}
DTYPE=${DTYPE:-float64}
N_D=${N_D:-200}
N_FD=${N_FD:-200}
N_REPEATS=${N_REPEATS:-1}
SHARDS=${SHARDS:-1}
# Space-separated agenda sizes to merge.  The default preserves the original
# two-agenda workflow; set ISSUES=5 for the main-paper-only production run.
ISSUES=${ISSUES:-"5 100"}

module load python
cd "$SLURM_SUBMIT_DIR"

for P in $ISSUES; do
    python3 cluster/sweep_one.py \
        --issues "$P" \
        --preset "$PRESET" \
        --n-agents "$N_AGENTS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --n-d "$N_D" \
        --n-fd "$N_FD" \
        --repeats "$N_REPEATS" \
        --n-shards "$SHARDS" \
        --merge \
        --dtype "$DTYPE"
done
