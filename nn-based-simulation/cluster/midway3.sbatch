#!/bin/bash
# One caslake node per agenda size, as a two-task array.
#
#   sbatch --account=pi-YOURPI cluster/midway3.sbatch
#
# Override the society size and resolution from the command line, e.g.
#   sbatch --account=pi-YOURPI --export=ALL,N_AGENTS=100 cluster/midway3.sbatch
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

# The two agenda sizes the phase-diagram figures need: preset.p_small and
# preset.p_large.  Array task 0 runs P=5, task 1 runs P=100.
ISSUES=(5 100)
P=${ISSUES[$SLURM_ARRAY_TASK_ID]}

PRESET=${PRESET:-full}
N_AGENTS=${N_AGENTS:-200}
BATCH_SIZE=${BATCH_SIZE:-256}
DTYPE=${DTYPE:-float64}

module load python

# The inner loop is single-threaded and memory-bandwidth bound; sweep.py already
# pins the BLAS pools to one thread per worker, and this keeps Slurm's view
# consistent with that.
export OMP_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID P=$P N=$N_AGENTS preset=$PRESET"
python3 cluster/sweep_one.py \
    --issues "$P" \
    --preset "$PRESET" \
    --n-agents "$N_AGENTS" \
    --batch-size "$BATCH_SIZE" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --dtype "$DTYPE"
