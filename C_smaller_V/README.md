# C << V Limit

This folder studies the singular hierarchy

```text
C / V = epsilon << 1
```

in the EDNNA discrimination model from `Discrimination2025(1).pdf`.

This is not the same limit as the existing `small-cv-phase-diagram/` folder. That
folder takes both uncertainties small with fixed ratio. Here the affective sector
is parametrically faster than the ideological sector.

## Main Scaling

For one receiver-emitter interaction,

```text
Delta w  = (F_w / gamma_C) sigma_e C x
Delta C  = (F_C / gamma_C^2) C x x^T C
Delta mu = (F_mu / gamma_V) V
Delta V  = (F_V / gamma_V^2) V^2
```

so, for `C = epsilon V Cbar` with `epsilon << 1`,

```text
Delta w / Delta mu = O(epsilon)
Delta C / Delta V  = O(epsilon^2)
```

up to modulation-function factors. The leading picture is therefore:

1. opinions `w` and overlaps `rho_ij` are frozen on the affective time scale;
2. trust variables `mu_{e|r}` rapidly adapt to the current agreement field
   `H = sigma_e w_r.x + D_{e|r}`;
3. only after trust has nearly saturated does the residual ideological drift act.

The natural slow-time variable is the number of encounters times `V`; ideological
motion appears only at order `epsilon`.

## Candidate Simplifying Regimes

### 1. Frozen-Ideology Trust Closure

Set `w` fixed and evolve only `mu`. The trust update is scalar per directed pair:

```text
mu_{e|r}(t+1) = mu_{e|r}(t) + V F_mu(H_{e|r}, mu_{e|r})
```

where `H_{e|r}` is determined by the fixed opinion agreement plus the
discrimination field. This gives a closed fast subsystem. The phase behavior of
`R_muc` can be approximated without simulating neural weights.

Interesting consequence: in the large positive discrimination-field regime,
`H` is dominated by class, so the fast trust dynamics already creates
class-correlated distrust before ideology moves. This is a clean route to the
paper's phase IV: high trust-class correlation with weak opinion-class
correlation.

### 2. Fast Trust Saturation, Then Signed Couplings

After fast relaxation, `eta_{e|r} = 1 - 2 Phi(mu_{e|r})` is close to a sign
function of the frozen field. Then the slow ideological update is approximately

```text
Delta w_r ~= C sigma_e F_w(H_{e|r}, mu^*_{e|r}) x.
```

Since `F_w` changes sign with trust, the slow dynamics becomes a perceptron-like
learning rule with quasi-static signed couplings:

```text
trusted emitter    -> learn with the emitter
distrusted emitter -> learn against the emitter
```

This may reduce the late-time ideological problem to a signed-network alignment
problem whose coupling signs are set by the fast trust layer.

### 3. Strong-Field Class Locking

If `|D| >> |sigma_e w_r.x|`, then

```text
H_{e|r} ~= D_{e|r}.
```

The affective dynamics becomes almost independent of issue geometry and agenda
size. In the symmetric discrimination case,

```text
D = d [[+1, -1], [-1, +1]]
```

with the sign convention used by this repository, same-class and cross-class
pairs are pushed into opposite trust basins. This should collapse the
discriminatory transition onto a mostly affective threshold in `d` and `f_d`.

### 4. Weak-Field Linear Response

For small `D`, expand

```text
F_mu(h + D, mu) = F_mu(h, mu) + D partial_h F_mu(h, mu) + O(D^2).
```

Because `w` is frozen on the fast scale, the first-order class signal in
`R_muc` is a response of the trust layer to the distribution of agreement fields
`h = sigma_e w_r.x`. This gives a possible analytic boundary estimate:
discrimination appears when the `D`-biased fast trust drift beats the initial
random trust dispersion.

### 5. Agenda-Complexity Separation

The paper argues that simple agendas and complex agendas differ by which
correlation establishes first. In `C << V`, the answer is structurally biased:
affective correlations establish first for any agenda because ideology is slow.
Agenda complexity enters later through the frozen distribution of agreement
fields and through the order-`C/V` ideological drift.

This is potentially useful: if simulations in this limit still show the
simple/complex reversal, then it cannot be explained only by the relative
learning rates of `w` and `mu`; if the reversal disappears, `C/V` is a control
parameter for the hate-first versus disagreement-first mechanism.

## Suggested Next Experiments

1. Implement a `fast_mu_frozen_w` mode that freezes `w,C,V` and updates only
   `mu`.
2. Compare its `R_muc(d,f_d)` heatmap to full dynamics at ratios
   `V/C = 10, 100, 1000`.
3. Measure the early-time slope of `R_muc` and `R_cw`. The prediction is
   `dR_cw/dtau = O(C/V)` while `dR_muc/dtau = O(1)`.
4. Fit a strong-field transition curve using only the distribution of class
   pair exposures and discriminatory-receiver fraction.

