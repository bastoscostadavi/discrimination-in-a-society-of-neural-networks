import numpy as np

from smallcv.society import SocietyBatch


def paired(c, v, dynamics):
    return SocietyBatch(
        n_agents=8,
        n_dim=6,
        n_issues=3,
        d=[-0.5, 0.5],
        f_d=[0.75, 0.75],
        initial_c=c,
        initial_v=v,
        seed=19,
        dynamics=dynamics,
    )


def snapshot(batch):
    return tuple(arr.copy() for arr in (batch.w, batch.C, batch.mu, batch.V))


def max_state_distance(a, b):
    return max(float(np.max(np.abs(x - y))) for x, y in zip(snapshot(a), snapshot(b)))


def test_small_cv_keeps_uncertainties_finite_and_updates():
    b = paired(0.05, 0.05, "small_cv")
    before_w = b.w.copy()
    before_C = b.C.copy()
    before_mu = b.mu.copy()
    before_V = b.V.copy()
    b.run(200)
    assert np.max(np.abs(b.w - before_w)) > 0.0
    assert np.max(np.abs(b.mu - before_mu)) > 0.0
    assert np.max(np.abs(b.C - before_C)) > 0.0
    assert np.max(np.abs(b.V - before_V)) > 0.0
    assert np.all(b.V > 0.0)


def test_full_and_reduced_single_interaction_converge_as_c_v_decrease():
    gaps = []
    for c in (0.1, 0.01):
        full = paired(c, c, "full")
        reduced = paired(c, c, "small_cv")
        x = full.X[0].copy()
        full.interact(0, 1, x)
        reduced.interact(0, 1, x)
        gaps.append(max_state_distance(full, reduced))
    assert gaps[1] < gaps[0]


def test_gamma_diagnostics_are_small_in_small_cv_regime():
    b = paired(0.01, 0.02, "small_cv")
    b.run(50)
    diag = b.gamma_diagnostics()
    assert diag["max_gamma_C_minus_1"].max() < 0.01
    assert diag["max_gamma_V_minus_1"].max() < 0.011
