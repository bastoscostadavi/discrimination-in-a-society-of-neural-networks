# Derivation Notes: Fast Affective Limit

## Starting Point

The paper's microscopic update for a receiver `r` listening to emitter `e` is

```text
H      = sigma_e (w_r.x) / gamma_C + D_{e|r}
h_mu  = mu_{e|r} / gamma_V

gamma_C = sqrt(1 + x.C_r.x)
gamma_V = sqrt(1 + V_{e|r})
```

with evidence

```text
Z(H,h_mu) = Phi(H) + Phi(h_mu) - 2 Phi(H) Phi(h_mu)
```

and modulation functions

```text
F_w  = (1 - 2 Phi(h_mu)) phi(H) / Z
F_mu = (1 - 2 Phi(H))    phi(h_mu) / Z.
```

The relevant increments are

```text
Delta w  = O(C F_w)
Delta mu = O(V F_mu)
Delta C  = O(C^2 F_C)
Delta V  = O(V^2 F_V).
```

Let

```text
C = epsilon V Cbar,       epsilon << 1.
```

On the fast time `s = t V`, the leading equations are

```text
d mu_{e|r} / ds = F_mu(H_{e|r}, mu_{e|r}) + O(V, epsilon)
d w_r       / ds = epsilon Cbar_r x sigma_e F_w(H_{e|r}, mu_{e|r}) + O(epsilon V)
```

so `w` is frozen at leading order.

## Fast Trust Flow

With `w` frozen, `H` is a constant for a selected directed pair and issue. Since
`phi(mu) > 0` and `Z > 0`,

```text
sign F_mu(H,mu) = sign(1 - 2 Phi(H)) = -sign(H)
```

for `H != 0`.

Therefore:

```text
H > 0  ->  mu decreases  -> eta = 1 - 2 Phi(mu) -> +1  (trust)
H < 0  ->  mu increases  -> eta = 1 - 2 Phi(mu) -> -1  (distrust)
H = 0  ->  F_mu = 0      -> no leading affective drift
```

The leading fast subsystem is not a finite fixed-point relaxation. It is a
saturation process with separatrix

```text
sigma_e (w_r.x) + D_{e|r} = 0.
```

This is the most useful simplification in the `C << V` limit.

## Quasi-Static Trust Closure

After fast saturation,

```text
eta_{e|r} ~= sign(H_{e|r})
```

away from the separatrix. Since

```text
eta = 1 - 2 Phi(mu)
```

the order parameter

```text
R_muc = < kappa_I kappa_J (eta_{I|J} + eta_{J|I}) >
```

can be approximated from the frozen distribution of fields `H_{e|r}` alone.

For the symmetric case-6 discrimination matrix used in the repository,

```text
D_{e|r} = d kappa_r kappa_e
```

for discriminatory receivers, and `D = 0` for non-discriminatory receivers.
Ignoring residual ideological correlations, the strong-field prediction is:

```text
same class:  H ~= +d  -> trust
cross class: H ~= -d  -> distrust
```

for discriminatory receivers. Thus the direct contribution to `R_muc` grows
roughly with `f_d`, before any opinion-class correlation appears.

## Slow Ideological Drift After Saturation

In the saturated trust layer,

```text
mu -> -infty for H > 0:
F_w -> phi(H) / Phi(H)

mu -> +infty for H < 0:
F_w -> -phi(H) / (1 - Phi(H)).
```

So the slow opinion rule becomes

```text
Delta w_r ~= C sigma_e x M(H_{e|r})
```

where

```text
M(H) =  phi(H) / Phi(H)       if H > 0
M(H) = -phi(H) / (1-Phi(H))   if H < 0.
```

The sign of `H` is therefore also the sign of the effective coupling. Since
`H = sigma_e w_r.x + D`, discrimination shifts the separatrix from `h = 0` to
`h = -D`.

This gives a compact interpretation:

```text
D > 0 for same-class pairs expands the learn-with-emitter basin.
D < 0 for cross-class pairs expands the learn-against-emitter basin.
```

## Regime Map Worth Testing

### Affective-Only Discrimination

Conditions:

```text
C/V << 1,     |D| comparable to or larger than typical |sigma_e w_r.x|.
```

Prediction:

```text
R_muc high, R_cw near initial value.
```

This is a clean phase-IV candidate: class-locked trust without ideological
class sorting.

### Delayed Ideological Sorting

Conditions:

```text
C/V << 1,     run time long enough for O(C) drift.
```

Prediction:

```text
R_muc rises first;
R_cw follows on a time scale larger by V/C.
```

This can be measured by early slopes.

### Separatrix-Dominated Weak Field

Conditions:

```text
|D| smaller than the width of the frozen agreement-field distribution.
```

Prediction:

Only pairs with `|sigma_e w_r.x| = O(|D|)` flip trust basin. The first-order
class signal is controlled by the density of frozen agreement fields near zero.

### Trust-Only Spin Glass

Conditions:

```text
D < 0 or reverse-discrimination cases, with C/V << 1.
```

Prediction:

The fast trust layer can become frustrated while the opinion layer remains
nearly random. This isolates an affective spin-glass-like regime from the
later ideological polarization.

