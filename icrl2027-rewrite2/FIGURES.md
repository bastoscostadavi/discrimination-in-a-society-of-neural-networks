# Figure plan

Every figure slot in the skeleton, mapped to output that already exists in
`../nn-based-simulation/figures/iclr/` or marked as a new experiment. Nothing is
copied; `main.tex` resolves through `\graphicspath`.

## Main text

| slot | section | source file | status |
|---|---|---|---|
| `fig:flows` | §3 rule | `learning_flows.pdf` | **exists.** Three panels at `D = -d, 0, +d`. Carries both mechanism properties at once; the best single figure in the paper. |
| `fig:phase` | §6 phase | `phase_diagram.pdf` | **exists.** The headline. Confirm the `d` axis matches the sign convention of `app_variants` before anything else. |
| `fig:correlations` | §6 phase | `correlation_maps.pdf` | **exists.** Two rows, low/high `α`. The `R_cw` panel is what separates regions III and IV. |
| `fig:frustration` | §6 phase | `frustration_maps.pdf` | **exists.** Needs axis relabel `B_I → B_ρ`, `B_A → B_η`. |
| — boundary `d_c(f_d)` | §6.1 | — | **new.** Extract at fixed `R_μc` threshold, error bars over seeds, finite-size collapse over `N ∈ {20,40,80}`. |
| — ablation grid | §7 | — | **new.** 5 panels (unablated, A0, A1, A2, A3) sharing the `fig:phase` colour scale. Spec in `app_ablation_spec.tex`. |
| `fig:trajectories` | §8 agenda | `agenda_trajectories.pdf` | **exists.** Needs the same axis relabel, and `Δt` stated in the caption. |

## Appendix

| slot | source file | status |
|---|---|---|
| `fig:modulation` | `modulation_contours.pdf` | **exists.** Axes currently show unscaled fields; relabel to `h_w`, `h_μ`. The source draft flags this too and never fixes it. |
| `fig:signs` | `sign_convention_comparison.pdf` | **exists.** Keep — a reader coming from the source draft needs it. |
| `D_1`–`D_5` variants | — | **new.** Correlation maps for the five asymmetric perturbations. |
| `Δt` stability | — | **new.** One small panel sweeping `Δt` over a decade, showing region assignment unchanged. |

## Not used

`modulation_surfaces.pdf`, `modulation_slices.pdf`, `order_parameter_maps.pdf`,
`phase_diagram_large_agenda.pdf`, `draft_comparison.pdf`.

Three views of the same four modulation functions is more than the paper needs —
pick one for the appendix. `order_parameter_maps` (2×5 grid) is a superset of
`correlation_maps` + `frustration_maps`; it is the better appendix figure if a
single omnibus panel is wanted, but it is too dense for the main text.
`phase_diagram_large_agenda` is subsumed by the two rows of `correlation_maps`
unless §8 needs a second headline map. `draft_comparison` is a validation artefact
— useful in the repository, not in the paper.

## Before submission

1. **Settle the `d` sign** and confirm every figure's `x` axis agrees with
   `app_variants.tex`. Highest priority; it orients the main result.
2. **Relabel** `B_I/B_A → B_ρ/B_η` and the modulation axes to `h_w/h_μ`.
3. **Re-run at ≥5 seeds.** Present maps are visibly single-realisation.
4. **Build the ablation grid.** This is the figure the paper is missing.
