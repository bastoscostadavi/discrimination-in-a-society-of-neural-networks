"""The planar reduction: what it preserves, and the gauge it fixes."""

import numpy as np
import pytest

from ednna.reduction import project, unit_rows


def rotation(n, seed=0):
    """A random orthogonal matrix in ``n`` dimensions."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))
    return q


def two_camp_trust(n_agents, kappa, sign=+1.0):
    """Trust rows for a population split along the class label.

    ``sign = +1`` is the discriminatory configuration, every agent trusting its own
    class; ``sign = -1`` is the frustrated one, every agent trusting the other.
    """
    rows = sign * np.outer(kappa, kappa)
    np.fill_diagonal(rows, 0.0)
    return rows


@pytest.fixture
def kappa():
    return np.where(np.arange(20) < 10, +1.0, -1.0)


def test_captured_is_one_when_the_population_is_already_planar():
    """A population spanning two directions loses nothing to a two-plane."""
    rng = np.random.default_rng(0)
    basis = rotation(9, seed=1)[:2]
    vectors = rng.normal(size=(30, 2)) @ basis
    _, captured, _ = project(vectors, 2)
    assert captured == pytest.approx(1.0)


def test_captured_separates_a_diffuse_population_from_a_structured_one():
    """The number the frustrated column of the portraits figure turns on.

    A plane cannot show much of an isotropic population in thirty dimensions, and
    shows almost all of one that has collapsed onto two directions.  The gap is what
    licenses reading a small percentage as "this sector has no structure", so it is
    worth pinning; the diffuse value sits above the naive ``2/D`` because the leading
    empirical singular values of a finite sample are inflated.
    """
    rng = np.random.default_rng(1)
    _, diffuse, _ = project(rng.normal(size=(400, 30)), 2)
    structured = project(rng.normal(size=(400, 2)) @ rotation(30, seed=6)[:2], 2)[1]
    assert 2 / 30 < diffuse < 0.15
    assert structured > 0.99


def test_coordinates_are_unchanged_by_a_rotation_of_the_ambient_space():
    """The reduction sees the population's own geometry, not its coordinates.

    Rotating every agent's vector by the same orthogonal matrix rotates the singular
    vectors with it, so the picture must come out identical.  If it did not, the
    figure would be reporting the arbitrary basis the simulation happened to use.
    """
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(25, 12))
    coords, captured, _ = project(vectors, 2)
    turned, captured_turned, _ = project(vectors @ rotation(12, seed=3), 2)
    assert captured_turned == pytest.approx(captured)
    assert np.allclose(turned, coords, atol=1e-9)


def test_lengths_never_exceed_one_and_match_the_captured_share(kappa):
    """Arrow length is a fraction of a unit vector, which the figure's scale assumes."""
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(20, 8)) + 3.0 * np.outer(kappa, rng.normal(size=8))
    coords, captured, _ = project(vectors, 2)
    lengths = np.linalg.norm(coords, axis=1)
    assert lengths.max() <= 1.0 + 1e-9
    assert np.mean(lengths**2) == pytest.approx(captured)


def test_the_reference_is_rotated_onto_the_first_axis(kappa):
    """The gauge that makes two panels comparable: the reference points along +x."""
    rows = two_camp_trust(20, kappa) + 0.05 * np.random.default_rng(4).normal(size=(20, 20))
    _, _, ref = project(rows, 2, reference=kappa)
    assert ref[0] > 0
    assert ref[1] == pytest.approx(0.0, abs=1e-9)


def test_the_reference_gauge_separates_trusting_your_own_class_from_the_reverse(kappa):
    """The one distinction the picture cannot make without the reference.

    Both configurations put trust on a single class-pure axis and are identical up
    to a sign, which is exactly what an unreferenced projection leaves free.  With
    the class indicator carried through, class $A$ lands on the positive side when
    agents trust their own class and on the negative side when they trust the other.
    """
    for sign in (+1.0, -1.0):
        coords, captured, ref = project(two_camp_trust(20, kappa, sign), 2,
                                        reference=kappa)
        # Not exactly 1: the self-entry is dropped, and it sits at a different
        # index in every row, which is a rank the two camps do not account for.
        assert captured > 0.9
        assert np.linalg.norm(ref) == pytest.approx(1.0)
        own_side = np.sign(coords[:, 0]) * kappa
        assert np.all(own_side == sign)


def test_without_a_reference_the_positive_class_fixes_the_first_axis(kappa):
    """The weaker gauge, and that it is applied rather than left to numpy."""
    rng = np.random.default_rng(5)
    vectors = np.outer(kappa, rng.normal(size=10)) + 0.1 * rng.normal(size=(20, 10))
    coords, _, _ = project(vectors, 2, positive_class=kappa == +1)
    assert coords[kappa == +1, 0].mean() >= 0


def test_unit_rows_survives_a_zero_row():
    """An agent with no trust profile at all must not produce a NaN panel."""
    out = unit_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out[1]) == pytest.approx(1.0)
