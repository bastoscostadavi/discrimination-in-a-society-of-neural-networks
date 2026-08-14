# Paper Reading Notes

Source: `../Discrimination2025(1).pdf`, dated March 2, 2026.

## Model Elements Used Here

- Agents carry perceptron weights `w_I`, covariance `C_I`, directed distrust
  means `mu_{J|I}`, and distrust variances `V_{J|I}`.
- An interaction selects receiver `r`, emitter `e`, and issue `x`; the emitter
  reports `sigma_e = sign(w_e.x)`.
- The receiver computes two fields:

```text
h_w  = sigma_e w_r.x / sqrt(1 + x.C_r.x)
h_mu = mu_{e|r} / sqrt(1 + V_{e|r})
```

- For discriminatory receivers, the opinion field is shifted:

```text
H = h_w + D_{e|r}.
```

- The evidence is

```text
Z = Phi(H) + Phi(h_mu) - 2 Phi(H) Phi(h_mu).
```

- The four modulation functions are `F_w`, `F_C`, `F_mu`, and `F_V`, with
  `F_w` controlling opinion updates and `F_mu` controlling trust updates.

## Relevant Interpretation From The Draft

The draft emphasizes that learning is driven by dissonance:

```text
agree with distrusted emitter
disagree with trusted emitter
```

The modulation functions then decide whether the receiver changes opinion or
changes trust. The draft also states that when the opinion field is large, the
trust sector is adjusted, while when the distrust field dominates, the opinion
sector is blamed for the surprise.

For the `C << V` limit, this becomes sharper: the trust sector is not only
preferred by the modulation structure, it also has the larger step size.

## Order Parameters To Track

The paper's phase diagram is organized by:

```text
R_wmu  opinion-trust correlation
R_muc  trust-class correlation
R_cw   opinion-class correlation
B_I    ideological balance
B_A    affective balance
```

For this limit the most diagnostic early-time signature is:

```text
R_muc changes on the fast time scale;
R_cw changes only on the slow time scale C/V.
```

That is why the limit is promising: it separates the paper's two discriminatory
regions into an early affective class-locking mechanism and a later ideological
sorting mechanism.

