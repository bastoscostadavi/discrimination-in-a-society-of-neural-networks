"""Figure-layout invariants that are easy to get silently wrong."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from ednna.plotting import phase_map, rgb_composite, use_style  # noqa: E402


@pytest.fixture(autouse=True)
def _style():
    use_style("paper")
    yield
    plt.close("all")


def test_phase_maps_put_zero_fd_at_the_bottom():
    """f_d increases upwards, and the data must follow the axis.

    Row 0 of a sweep is f_d = 0, so the orientation is carried by
    ``origin="lower"`` and by the extent, which have to agree: getting one
    without the other flips every map upside down, and the mistake is invisible
    unless the data is asymmetric in f_d.
    """
    d = np.linspace(-1, 1, 8)
    fd = np.linspace(0, 1, 8)
    data = np.tile(fd[:, None], (1, 8))  # value equals f_d
    fig, ax = plt.subplots()
    im = phase_map(ax, data, d, fd, "R_wmu", colorbar=False)
    bottom, top = ax.get_ylim()
    assert bottom < top, "f_d must increase upwards"
    assert im.origin == "lower", "row 0 (f_d = 0) must be drawn at the bottom"
    assert im.get_array()[0, 0] == pytest.approx(fd[0])


def test_phase_map_extent_matches_the_data_range():
    d = np.linspace(-1, 1, 5)
    fd = np.linspace(0, 1, 5)
    fig, ax = plt.subplots()
    im = phase_map(ax, np.zeros((5, 5)), d, fd, "R_muc", colorbar=False)
    left, right, bottom, top = im.get_extent()
    assert (left, right) == (-1.0, 1.0)
    assert (bottom, top) == (0.0, 1.0)


def test_diverging_parameters_are_centred_on_zero():
    """R_muc and B_eta are signed; white must sit at 0, not at the midpoint."""
    d = np.linspace(-1, 1, 4)
    fd = np.linspace(0, 1, 4)
    fig, ax = plt.subplots()
    im = phase_map(ax, np.linspace(-1, 1, 16).reshape(4, 4), d, fd, "R_muc", colorbar=False)
    assert im.norm(0.0) == pytest.approx(0.5, abs=1e-9)


def test_rgb_composite_encodes_the_four_regions():
    """Region colours are what the phase diagram's caption claims."""
    zeros = np.zeros((1, 1))
    ones = np.ones((1, 1))

    # (I) reverse discrimination: R_muc at its floor, nothing else -> black
    black = rgb_composite(-ones, zeros, zeros)[0, 0]
    np.testing.assert_allclose(black, [0, 0, 0], atol=1e-9)

    # (II) neutral: only opinion-trust correlated -> blue
    blue = rgb_composite(zeros, zeros, ones)[0, 0]
    np.testing.assert_allclose(blue, [0.5, 0, 1], atol=1e-9)

    # (IV) class-only discrimination: trust-class and opinion-trust, no
    # opinion-class -> magenta
    magenta = rgb_composite(ones, zeros, ones)[0, 0]
    np.testing.assert_allclose(magenta, [1, 0, 1], atol=1e-9)

    # (III) ideological discrimination: all three -> pale
    pale = rgb_composite(ones, ones, ones)[0, 0]
    np.testing.assert_allclose(pale, [1, 1, 1], atol=1e-9)


def test_rgb_composite_clips_out_of_range_values():
    big = np.full((1, 1), 5.0)
    small = np.full((1, 1), -5.0)
    rgb = rgb_composite(big, small, big)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
