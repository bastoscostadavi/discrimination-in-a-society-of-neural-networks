"""The panels have to draw.

A figure script that crashes is cheap to find out about now and expensive to
find out about after an hour of measurement, so the three panels are exercised
here on synthetic values with the shape the real ones have.
"""

import matplotlib
import numpy as np
import pytest
from ednna.modulation import F_mu, F_w

matplotlib.use("Agg")

from llmmod2 import plotting  # noqa: E402


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    h_w = rng.uniform(-2.5, 2.5, 240)
    h_mu = rng.uniform(-2.0, 2.0, 240)
    d_w = 0.4 * F_w(h_w, h_mu) + rng.normal(0, 0.05, 240)
    d_mu = 0.4 * F_mu(h_w, h_mu) + rng.normal(0, 0.05, 240)
    return h_w, h_mu, d_w, d_mu


def test_fit_scale_recovers_a_known_scale(sample):
    h_w, h_mu, d_w, _ = sample
    assert plotting.fit_scale(h_w, h_mu, d_w, F_w) == pytest.approx(0.4, rel=0.1)


def test_the_three_panels_draw(sample, tmp_path):
    h_w, h_mu, d_w, d_mu = sample
    plotting.use_style("iclr")
    fig, axes = plotting.plt.subplots(1, 3, figsize=plotting.panel(1.0, 0.36))
    plotting.plane_panel(axes[0], h_w, h_mu, d_w, F_w, 0.4)
    plotting.gate_panel(axes[1], h_w, h_mu, d_w, F_w, 0.4)
    r = plotting.crossover_panel(axes[2], h_w, h_mu, d_w, d_mu, F_w, F_mu, 0.4)
    assert np.isfinite(r)
    assert plotting.save(fig, "smoke", tmp_path).is_file()


def test_the_crossover_is_a_real_prediction(sample):
    """``F_w / F_mu`` is exactly one on the diagonal, whatever the field value.

    That is what makes the crossover panel worth drawing: its zero crossing is
    fixed by the theory and not by any fitted scale.
    """
    for h in (-1.7, -0.4, 0.9, 2.1):
        assert F_w(h, h) == pytest.approx(F_mu(h, h))
    # and away from it the less certain sector is the one that moves
    assert abs(F_w(1.5, 0.0)) < abs(F_mu(1.5, 0.0))
    assert abs(F_w(0.0, 1.5)) > abs(F_mu(0.0, 1.5))


def test_gate_panel_survives_an_empty_arm(sample):
    """A sweep that never produced a disagreeing message must still draw."""
    h_w, h_mu, d_w, _ = sample
    plotting.use_style("iclr")
    fig, ax = plotting.plt.subplots()
    plotting.gate_panel(ax, np.abs(h_w), h_mu, d_w, F_w, 0.4)
    plotting.plt.close(fig)
