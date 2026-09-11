# Does IIIb eventually become IIIa?

This directory is a self-contained lifetime experiment for the manuscript's
simple-agenda discriminatory regimes. Start with **REPORT.md** for the findings.
The original manuscript and simulation folders are not modified.

## Reproduce

Requires Python 3.10+, a C++17 compiler named `clang++`, and:

```sh
python3 -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python3 experiment.py --validate
python3 experiment.py --issues 5 --max-time 100000 --seeds 12 --workers 4
python3 experiment.py --issues 100 --max-time 10000 --seeds 12 --workers 4
python3 experiment.py --issues 5 --max-time 100000 --seeds 12 --points b_reference --agents 60 --workers 2
python3 probes.py
python3 mechanism.py
python3 analyze.py
python3 complete_report.py
```

Run commands from this directory. There are no imports or data dependencies on
sibling directories. `experiment.py` compiles its local `kernel.cpp` automatically.
The existing results are cached; a longer requested duration reruns a baseline
from its original seed, retaining measurements along the trajectory. Delete only
the relevant `results/N*.json` files to force fresh baseline runs. `probes.py`
resumes the saved N=40, P=5, IIIb states at exactly 100,000 and must run after that
baseline; it writes separate files and leaves baseline results intact.

## Protocol

- Main size N=40; opinion dimension K=30; P=5 and P=100.
- Original priors, discrimination field sign, Bernoulli assignment of biased
  agents, and double-precision numerical safeguards are retained.
- Seven parameter points: (b,f)=(0,0), (0.5,1), and b=1 with
  f in {0.25,0.5,0.75,0.9,1}. The paper's portrait references are
  IIIa=(0.5,1) and IIIb=(1,0.5).
- Twelve independent initial-condition and schedule seeds, 202609110–202609121.
  Seeds are reused across parameter points as common random numbers;
  repetitions at a given point are independent.
- Follow each society, without restarting it at observation times, through
  Δt=0,125,500,1000,2500,5000,10000 and, for P=5,
  25000,50000,100000. Δt means **mean interactions per ordered pair**:
  each run has Δt N(N−1) updates.
- At IIIb, continue all twelve P=5 societies to 1,000,000. In paired diagnostic
  branches, perturb each agenda opinion vector by 1% of its norm or reset only
  its agenda covariance to identity; continue those branches to 200,000.
- Repeat the IIIb baseline at N=60 through 100,000, with twelve seeds.
- Record full-space, agenda-space, and binary stated-opinion class correlations,
  opinion–trust and trust–class correlations, subgroup opinion alignment,
  uncertainties, covariance eigenvalues, and safeguard activation counts.
- Summaries use across-seed Student-t 95% confidence intervals. Changes are
  paired within seed. These intervals describe seed variation at the tested
  points, not uncertainty about infinite time or unsampled parameter regions.

The original phase map defines no numerical classification threshold. We report
continuous order parameters and microscopic diagnostics instead of assigning
new labels using an arbitrary threshold.

## Acceleration and checks

For isotropic initial covariance, the agenda span and its orthogonal complement
remain uncoupled: every update is in the agenda span. We use an orthonormal SVD
basis of that span, evolve the same equations in rank min(P,K), and retain each
agent's original orthogonal opinion vector exactly when reconstructing full-space
observables. This is an exact change of coordinates, not a reduced dynamical
approximation. For P=100 it is simply a rotation of all 30 dimensions.

`kernel.cpp` implements those same updates in compiled double precision, without
fast-math or an altered learning rate. The validation sends identical interaction
schedules through this kernel and the copied original NumPy implementation,
checking weights, full covariances, trust means, and variances at P=5 and P=100.
Results are in `results/validation.json`.

Schedules are generated in blocks of at most 100,000 interactions using NumPy's
Generator. Observation boundaries can split blocks, so changing the observation
grid changes the exact random schedule, though not its distribution. All paired
probe arms share their observation grid and saved random-generator state.

## Contents

- `REPORT.md`: findings, limits, and suggested manuscript language.
- `experiment.py`, `kernel.cpp`: reproducible baseline simulator.
- `probes.py`: actual checkpoint continuations and perturbation experiments.
- `mechanism.py`: conditional expected-drift and local-curvature diagnostic.
- `analyze.py`, `final_figures.py`, `complete_report.py`: summary tables, static figures, and the completed-results report section.
- `ednna/`: copied original simulation package, unchanged.
- `reference/paper-main.tex`: manuscript source snapshot for interpretation;
  figures and bibliography are not needed to run these experiments.
- `reference/provenance.json`: source commit, hashes, and environment.
- `results/*.json`: every baseline trajectory and configuration.
- `results/*.npz`: final microscopic states and RNG states.
- `results/probes/`: continuation and intervention trajectories/checkpoints.
- `results/trajectories.csv`, `results/summary.json`: tidy baseline data and
  paired confidence intervals.
- `figures/`: PNG and PDF plots. Logs record completed computations.

No manuscript text was changed. Finite runs and conditional stability diagnostics
cannot prove that IIIa and IIIb are distinct infinite-time, infinite-population
phases; the report distinguishes that claim from the finite-time evidence.
