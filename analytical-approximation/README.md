# Controlled Small-(C,V) Reduction

This folder implements Task 1: a controlled small-`C,V` reduction of the
microscopic entropic learning dynamics.  It is deliberately separate from
`small-cv-phase-diagram/`; this code stops at drift equations and
Gaussian-averaged kernels, and does not compute the `(f_d,d)` phase diagram.

The package keeps the manuscript modulation functions
`F_w`, `F_C`, `F_mu`, and `F_V` intact.  The small parameter is represented by

```text
C = epsilon * Cbar
V = epsilon * Vbar
tau = epsilon * t
```

so the leading dynamics are written for `w`, `mu`, `Cbar`, and `Vbar` on the
slow time `tau`.

## Contents

- `controlledcv/modulation.py`: exact evidence and modulation functions, using
  a cancellation-resistant evidence expression and no arbitrary evidence floor.
- `controlledcv/microscopic.py`: full one-interaction increments and
  leading-order slow-time increments.
- `controlledcv/fields.py`: reduced density `p(h | q_r, rho)`.
- `controlledcv/kernels.py`: quadrature and Monte Carlo implementations of
  affective, ideological, and covariance kernels.
- `tests/`: validation for modulation stability, single-interaction asymptotics,
  field-density normalization, and quadrature-vs-Monte Carlo agreement.

## Run

```bash
cd analytical-approximation
pip install -r requirements.txt
pytest
```

The interaction-rate convention is intentionally explicit: all kernels are
conditional on a selected ordered emitter-receiver pair.  Global factors such as
`1/N` or `1/[N(N-1)]` should be added only when assembling these pairwise drifts
into a population-level ODE.
