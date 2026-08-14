# Small-C,V Phase-Diagram Simulator

This folder is a standalone implementation of the neural-network society model in the small-uncertainty regime

```text
C << 1,   V << 1.
```

It is separate from the original `nn-based-simulation/` code and is intended for studying the collective phase diagram in the external parameter plane `(d, f_d)`, where `d` is the discrimination-field strength and `f_d` is the fraction of discriminatory receivers.

## What is approximated

The full microscopic model uses

```text
gamma_C = sqrt(1 + x^T C_r x)
gamma_V = sqrt(1 + V_{e|r})

h_w  = sigma_e (w_r · x) / gamma_C
h_mu = mu_{e|r} / gamma_V
```

In the small-`C,V` approximation, only the field normalizations are replaced by their leading values:

```text
gamma_C ≈ 1
gamma_V ≈ 1

h_w  ≈ sigma_e (w_r · x)
h_mu ≈ mu_{e|r}
```

For discriminatory receivers, the microscopic field entering the modulation functions is still shifted by the actual discrimination matrix entry:

```text
h_w^D = h_w + D_{e|r}
```

Rows of `D` index receiver class and columns index emitter class. Case 6 is the default, but the scripts expose the discrimination case as a parameter.

## What remains microscopic

This is not a phenomenological or Landau-style replacement model. The simulator keeps the entropic-learning modulation functions

```text
Z, F_w, F_C, F_mu, F_V
```

and updates the microscopic state variables:

```text
w_r      += F_w(h_w^D, h_mu) * C_r x * sigma_e
C_r      += F_C(h_w^D, h_mu) * C_r x x^T C_r
mu_{e|r} += F_mu(h_w^D, h_mu) * V_{e|r}
V_{e|r}  += F_V(h_w^D, h_mu) * V_{e|r}^2
```

The small parameters still matter because `C` and `V` set the learning scale:

```text
Delta w  = O(C)
Delta mu = O(V)
Delta C  = O(C^2)
Delta V  = O(V^2)
```

So the code does **not** set `C=V=0`; it initializes

```text
C_I(0) = c I_K
V_{J|I}(0) = v
```

and usually studies fixed ratio `r = v/c`.

## Outputs

For each `(d, f_d)` grid point and repeat, the code simulates a full society and measures:

- `R_wmu`: opinion-trust correlation
- `R_muc`: trust-class correlation, the main discrimination order parameter
- `R_cw`: opinion-class correlation
- `B_I`, `B_A`: ideological and affective balance diagnostics
- maximum observed `|gamma_C - 1|` and `|gamma_V - 1|` diagnostics

Heatmaps and phase-composite figures are written under `figures/`. Raw sweep arrays are written under `data/`.

## Running a sweep

Example coarse sweep:

```bash
python3 scripts/run_phase_diagram.py   --n-d 21 --n-fd 21 --repeats 2   --c 0.05 --ratio 1 --steps-at-c1 50 --batch-size 32
```

True 200x200 sweep currently being used:

```bash
python3 scripts/run_phase_diagram.py   --n-d 200 --n-fd 200 --repeats 1   --c 0.05 --ratio 1 --steps-at-c1 50 --batch-size 16
```

The number of microscopic interactions per society is

```text
steps = steps_at_c1 * N * (N - 1) / c.
```

With `N=40`, `c=0.05`, and `steps_at_c1=50`, this is 1,560,000 microscopic interactions per society run.

## Checkpointing and resume

Sweeps save one checkpoint per batch under

```text
data/checkpoints/
```

If a run is interrupted, rerun the same command without `--no-cache`. Completed checkpoints are reused and only missing batches are simulated. Smaller `--batch-size` values save progress more frequently; larger values are usually faster but risk losing more in-progress work if interrupted.

## Validation

The test suite checks:

- modulation-function formulas,
- sector symmetries,
- nonzero finite-`C,V` learning,
- convergence of one full microscopic interaction to the reduced interaction as `c,v` decrease.

Run:

```bash
python3 -m pytest -q
```

A short ICLR-style description of the implementation and exploratory results is in:

```text
paper/iclr_short_paper.tex
paper/iclr_short_paper.pdf
```
