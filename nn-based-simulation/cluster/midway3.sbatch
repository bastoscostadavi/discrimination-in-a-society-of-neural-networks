#!/bin/bash
# One caslake node per agenda/shard pair.  With SHARDS=1 this is the original
# two-task array; with SHARDS=2, tasks 0-1 run P=5 and tasks 2-3 run P=100.
#
#   sbatch --account=pi-YOURPI cluster/midway3.sbatch
#
# Override the society size, resolution, and averaging from the command line,
# e.g. the N=100, 300x300, two-repeat main-paper run with lightweight snapshots,
# split into eight P=5 shards (array tasks 0 through 7):
#   sbatch --account=pi-YOURPI \
#     --array=0-7%8 --time=12:00:00 \
#     --export=ALL,N_AGENTS=100,N_D=300,N_FD=300,N_REPEATS=2,SHARDS=8,SAVE_SNAPSHOTS=1,BATCH_SIZE=16 \
#     cluster/midway3.sbatch
#
#SBATCH --job-name=ednna-sweep
#SBATCH --partition=caslake
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=36:00:00
#SBATCH --output=cluster/logs/%x-%A_%a.out
#SBATCH --error=cluster/logs/%x-%A_%a.out

set -euo pipefail

# The two agenda sizes the phase-diagram figures need.  Consecutive groups of
# SHARDS array tasks run each agenda.
ISSUES=(5 100)

PRESET=${PRESET:-full}
N_AGENTS=${N_AGENTS:-200}
BATCH_SIZE=${BATCH_SIZE:-256}
DTYPE=${DTYPE:-float64}
N_D=${N_D:-200}
N_FD=${N_FD:-200}
N_REPEATS=${N_REPEATS:-1}
SHARDS=${SHARDS:-1}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-0}

AGENDA_INDEX=$((SLURM_ARRAY_TASK_ID / SHARDS))
SHARD_INDEX=$((SLURM_ARRAY_TASK_ID % SHARDS))
if (( AGENDA_INDEX >= ${#ISSUES[@]} )); then
    echo "array task $SLURM_ARRAY_TASK_ID is outside 0..$((2 * SHARDS - 1))" >&2
    exit 2
fi
P=${ISSUES[$AGENDA_INDEX]}

module load python

# The inner loop is single-threaded and memory-bandwidth bound; sweep.py already
# pins the BLAS pools to one thread per worker, and this keeps Slurm's view
# consistent with that.
export OMP_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

SNAPSHOT_ARGS=()
if [[ "$SAVE_SNAPSHOTS" == "1" ]]; then
    SNAPSHOT_ARGS+=(--save-snapshots)
fi

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID P=$P shard=$((SHARD_INDEX + 1))/$SHARDS N=$N_AGENTS preset=$PRESET grid=${N_D}x${N_FD} repeats=$N_REPEATS snapshots=$SAVE_SNAPSHOTS"
python3 cluster/sweep_one.py \
    --issues "$P" \
    --preset "$PRESET" \
    --n-agents "$N_AGENTS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --n-d "$N_D" \
    --n-fd "$N_FD" \
    --repeats "$N_REPEATS" \
    --n-shards "$SHARDS" \
    --shard-index "$SHARD_INDEX" \
    --dtype "$DTYPE" \
    "${SNAPSHOT_ARGS[@]}"
