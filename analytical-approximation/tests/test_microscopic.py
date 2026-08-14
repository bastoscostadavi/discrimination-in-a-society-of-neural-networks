import numpy as np

from controlledcv.microscopic import full_increment, leading_increment


def random_state(seed=3):
    rng = np.random.default_rng(seed)
    k = 5
    w_r = rng.normal(size=k)
    w_e = rng.normal(size=k)
    A = rng.normal(size=(k, k))
    Cbar = A @ A.T / k
    x = rng.normal(size=k)
    mu = 0.35
    Vbar = 0.8
    D = -0.2
    return w_r, w_e, Cbar, mu, Vbar, x, D


def max_scaled_error(epsilon):
    w_r, w_e, Cbar, mu, Vbar, x, D = random_state()
    full = full_increment(w_r, w_e, epsilon * Cbar, mu, epsilon * Vbar, x, D=D)
    lead = leading_increment(w_r, w_e, Cbar, mu, Vbar, x, D=D)
    return max(
        np.max(np.abs(full.dw / epsilon - lead.dw)),
        np.max(np.abs(full.dC / (epsilon * epsilon) - lead.dC)),
        abs(full.dmu / epsilon - lead.dmu),
        abs(full.dV / (epsilon * epsilon) - lead.dV),
    )


def test_single_interaction_expansion_converges():
    err_large = max_scaled_error(0.05)
    err_small = max_scaled_error(0.005)
    assert err_small < 0.12 * err_large


def test_field_approximation_error_is_order_epsilon():
    w_r, w_e, Cbar, mu, Vbar, x, D = random_state()
    errors = []
    for epsilon in (0.05, 0.005):
        full = full_increment(w_r, w_e, epsilon * Cbar, mu, epsilon * Vbar, x, D=D)
        lead = leading_increment(w_r, w_e, Cbar, mu, Vbar, x, D=D)
        errors.append(max(abs(full.h_w - lead.h_w), abs(full.h_mu - lead.h_mu)))
    assert errors[1] < 0.12 * errors[0]
