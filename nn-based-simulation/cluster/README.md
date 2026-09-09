# Running the sweeps on a UChicago cluster

Two routes, with different hardware and very different walltime caps:
RCC Midway3 (open to any PI's group, 48-core/180 GB nodes, 36 h jobs) and
the DSI cluster (DSI affiliation required, 128-thread/1.5 TB CPU nodes,
**4 h jobs**). Midway3 first, then the DSI route and what its cap changes.

## 1. Get RCC access (this part only you can do)

RCC accounts hang off a PI's group account, so the order matters:

1. **Your PI needs an RCC PI account.** Faculty/staff eligible to be a grant PI
   request one at [PI Account Request](https://rcc.uchicago.edu/accounts-allocations/pi-account-request).
   It comes with free start-up service units — enough to port and test code
   before applying for a real allocation. Nothing here costs money.
2. **You request a user account** on your PI's group at
   [Request Account](https://rcc.uchicago.edu/accounts-allocations/request-account),
   naming their group. You need a UChicago CNetID.
3. **A larger allocation**, once the start-up units run out, via
   [Request Allocation](https://rcc.uchicago.edu/accounts-allocations/request-allocation).
   Free for PIs; the units exist to ration shared nodes, not to bill.

Then log in with your CNetID and CNet password:

```sh
ssh YOUR_CNETID@midway3.rcc.uchicago.edu
```

Optional `~/.ssh/config` entry, so `ssh midway3` is enough:

```
Host midway3
    HostName midway3.rcc.uchicago.edu
    User YOUR_CNETID
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

On the cluster, find the account name to charge (`pi-something`) and what you
have left:

```sh
rcchelp balance
rcchelp allocations
sinfo -o "%P %l %D %c %m"   # partitions, max walltime, nodes, cores, memory
```

## 2. Put the code there

Only the code needs to travel; `data/` is a cache the sweep rebuilds.

```sh
rsync -av --exclude data --exclude figures --exclude __pycache__ \
    ./ midway3:~/ednna/nn-based-simulation/

ssh midway3
cd ~/ednna/nn-based-simulation
module load python
python3 -m pip install --user -r requirements.txt
python3 -m pytest -q        # 77 tests, ~15 s: confirms the port
```

## 3. Submit

One caslake node (48 cores, 180 GB) per agenda size, as a two-task array:

```sh
sbatch --account=pi-YOURPI cluster/midway3.sbatch                        # N=200
sbatch --account=pi-YOURPI --export=ALL,N_AGENTS=100 cluster/midway3.sbatch
```

`cluster/sweep_one.py` runs exactly one `(d, f_d)` sweep and writes it under the
same cache tag the figure scripts read, so the array's two tasks are the two
sweeps `make_all.py` would otherwise do back to back in one process. Watch them
with `squeue -u $USER` and `tail -f cluster/logs/*.out`.

## 4. Bring the caches home and draw

```sh
rsync -av midway3:~/ednna/nn-based-simulation/data/ data/
python scripts/make_all.py --preset full --n-agents 200 \
    --batch-size 256 --workers 48
```

The `--batch-size` and `--workers` here are not doing any work locally --- they
are part of the cache filename, so they have to match what the job ran or the
laptop starts the 23-hour sweep over again. With them, every figure loads from
cache and nothing re-simulates. Figures are PDF only, into
`figures/{paper,iclr}/`.

## What a larger society costs

One interaction costs `O(K²)` regardless of `N`, and the run length is
`interactions_per_channel * N * (N - 1)`, so cost starts out scaling as **N²**
— the grid resolution is the cheap axis and `N` is the expensive one.

But that N² only holds while the batch stays wide enough. Batch state is
`3 * N² * R * 8` bytes for the three pairwise matrices (`mu`, `V`, `D`) plus
`N * R * K² * 8` for the covariances, so **the largest `R` that fits a worker
falls as 1/N²** — and `R` is what amortizes numpy's per-call overhead across
societies, since one interaction is `O(K²)` work on `K=30` arrays and otherwise
disappears under the call overhead.

The penalty is a cliff, not a slope. Measured per-society-step cost is flat at
~3–5 µs for `R` anywhere from 16 to 256 (bandwidth-bound, and `R=256` is no
better than `R=16`), then blows up as `R` approaches 1: at `N=10000` it is
19.6 µs at `R=4` and 44 µs at `R=1`. So what matters is only whether a worker
can still hold `R ≈ 16`, which is `384 * N²` bytes:

| node | RAM per worker | keeps `R ≈ 16` up to |
|------|----------------|----------------------|
| Midway3 caslake, 48 workers | 3.75 GB | `N ≈ 3100` |
| DSI `m`, 128 workers        | 12 GB   | `N ≈ 5600` |
| DSI `m`, 64 workers         | 23 GB   | `N ≈ 7800` |

Past that the effective exponent climbs from 2 toward 4.

Measured on one core in isolation, per sweep at `full` resolution (200×200);
derate by ~2× for 48 workers competing for memory bandwidth:

| N     | largest R per worker | core-hours | on one 48-core node | effective exponent |
|-------|---------------------|------------|---------------------|--------------------|
| 40    | 1024                | 43         | 0.9 h               | —      |
| 200   | 512                 | 600        | 12 h                | 2.1    |
| 400   | 256                 | 2,600      | 54 h                | 2.0    |
| 1000  | 64                  | 16,000     | 14 days             | 2.0    |
| 2000  | 16                  | 112,000    | 98 days             | 2.8    |
| 4000  | 4                   | 1.0M       | 2.4 years           | 3.2    |
| 10000 | 1                   | 32M        | 76 years            | 3.8    |

**Where the practical ceiling is.** `N=400` fits two 36 h jobs as they are
submitted. `N=1000` is ~16,000 core-hours per sweep: real but not one job —
it needs the grid sharded over ~20 nodes, which the cache layer does not do
yet, and an allocation to match. Past that it stops being a scheduling problem.

**`N=10000` is not reachable by adding nodes.** One society is 5.0e10
sequential interactions, and no amount of parallelism shortens it — the
interactions are ordered by construction, so a single grid point is a single
core for its whole duration. Measured, that point costs:

| node | `R` that fits | µs/society-step | one grid point | 200×200 sweep |
|------|---------------|-----------------|----------------|---------------|
| Midway3 caslake | 1 | 44 | 26 days | 25M core-h |
| DSI `m` (1.5 TB) | 8 | ~14 (extrapolated from 19.6 at `R=4`) | 8 days | 8M core-h |

The DSI node's 1.5 TB is worth ~3× here, entirely by keeping `R` off the
cliff — and it is still 900 core-years for the full grid, against a **4 h**
walltime cap. Getting to `N=10000` at all would need checkpoint/restart in
`sweep.py`, and getting there affordably would need an inner loop that is not
paying numpy overhead per interaction (the small-`R` step is ~10× off the
vectorized rate, which is what a compiled loop would recover).

If the question is a finite-size trend rather than a phase diagram, a 1D cut or
a handful of points is the way: each `N=10000` point is still 8 days, but ten of
them is ~2,000 core-hours instead of 8M.

## The DSI cluster route

Better hardware for this workload, worse walltime. Worth it if you are eligible.

**Eligibility** — you need one of: a DSI-related grant, DSI or DSI-affiliated
faculty status, or (the student path) **written approval from a DSI faculty
mentor**. It is not open to the University at large; without a DSI tie the
answer is RCC, above.

**Request** — email **techstaff@cs.uchicago.edu**, saying in the subject that it
concerns the *DSI cluster* (they run several), and include your CNetID, a short
note on your DSI connection, and your advisor on CC. They create the account and
reply.

**Log in** — `ssh YOUR_CNETID@login.ds.uchicago.edu`, a load balancer over three
frontends (`fe01`–`fe03`). The login nodes are capped at 1 CPU / 8 GB / 12 h per
process and are explicitly not for computation, so `sinteractive`-style work goes
to a compute node.

**The node you want is the `m` series**: 3 nodes, 128 CPU threads, 1.5 TB RAM,
*no GPUs*. That is the right shape for this code — it is pure numpy on CPU and
would leave an A100 idle — and the 1.5 TB is the interesting part, since RAM per
worker is what keeps `R` off the cliff (see the table above). Treat the 128 as
~64 useful for this workload: the inner loop is memory-bandwidth bound, which is
not something SMT helps. Partitions are granted through your advisor's group, so
ask techstaff which partition covers the `m` nodes when you request the account.

**The 4 h cap changes the plan.** Jobs are limited to 4 hours per partition, and
the docs say plainly that you are expected to write your code around it.
`midway3.sbatch` asks for 36 h and will not run there. Against the measured
numbers that means:

| N   | one sweep on ~64 cores | fits 4 h? |
|-----|------------------------|-----------|
| 40  | 0.7 h                  | yes       |
| 200 | 9 h                    | no        |
| 400 | 40 h                   | no        |

So on the DSI cluster **everything past `N≈100` needs the `(d, f_d)` grid
sharded into ≤4 h array tasks**, with the shards merged into one cache
afterwards. `sweep.py` does not do that yet: `cache_path` hashes the whole
configuration into a single file per sweep, with no notion of a partial grid.
It is a contained change — give `sweep()` an index range, write
`sweep_P5_200x200_<digest>.part<k>.npz`, and merge — but it has to happen before
the DSI cluster is usable for anything but `quick`. Ask me when you have the
account and know the partition.

## The cache-key trap

`cache_path` hashes the *whole* configuration, so `--workers` and
`--batch-size` are both part of the filename even though only `--batch-size`
changes the numbers. A cluster run at 48 workers therefore writes a different
file than a laptop run at 10 — it will not reuse your existing
`sweep_P5_200x200_c584209e3fa2.npz`, and your laptop will not reuse its result
unless you pass the same `--workers 48` when you plot. For the large-N runs this
is moot, since `N` changes the digest anyway, but pass the same flags to
`make_all.py` in step 4 as you passed to `sbatch` in step 3.
