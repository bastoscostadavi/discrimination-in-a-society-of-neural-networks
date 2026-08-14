"""The discrimination field and the controls that isolate it.

The paper's central claim is that the effect requires the perturbation to be
*correlated with class*, not merely present.  That claim is only as good as the
guarantee that the control fields really are matched in everything else, so the
matching is asserted here rather than described in prose.
"""

import numpy as np
import pytest

from socsim import FieldSpec, ModelConfig, SocietyBatch, field_matrix
from socsim.discrimination import CASES, build_field
from socsim.seeds import RunKey, point_id

N = 40


def _classes(n=N):
    c = np.zeros(n, dtype=np.int8)
    c[n // 2 :] = 1
    return c


def _build(spec, seed=0, class_of=None):
    cls = _classes() if class_of is None else class_of
    return build_field(
        spec,
        cls,
        np.random.default_rng(seed),
        np.random.default_rng(seed + 1000),
    )


# -- the sign convention ---------------------------------------------
def test_case_six_is_tolerant_to_the_in_group():
    """d > 0 must mean in-group tolerance, out-group intolerance.

    This is forced by the algorithm: F_mu carries (1 - 2 Phi(h_w)), which is
    negative for h_w > 0, and mu is distrust -- so raising h_w builds trust.
    The source material tabulates the opposite sign while placing the
    discriminatory regime at d > 0; the two cannot both hold.
    """
    D = field_matrix(0.5, case=6)
    assert D[0, 0] > 0 and D[1, 1] > 0
    assert D[0, 1] < 0 and D[1, 0] < 0


def test_compact_form_matches_the_template():
    """D_{e|r} = z_r d kappa_r kappa_e, with +d on the diagonal.

    The plan document writes this with a leading minus, which is the
    inconsistent reading; it would mirror the entire regime map in d.
    """
    d = 0.7
    kappa = np.array([1.0, -1.0])
    compact = d * np.outer(kappa, kappa)
    np.testing.assert_allclose(compact, field_matrix(d, case=6))


@pytest.mark.parametrize("case", CASES)
def test_cases_are_antisymmetric_in_d(case):
    np.testing.assert_allclose(field_matrix(0.4, case), -field_matrix(-0.4, case))


def test_legacy_convention_is_the_mirror_image():
    np.testing.assert_allclose(
        field_matrix(0.5, case=6, convention="legacy"), field_matrix(-0.5, case=6)
    )


def test_class_swap_invariance():
    """Renaming the classes must not change the physics (case 6 is symmetric)."""
    spec = FieldSpec(kind="class", case=6, d=0.6, f_d=1.0)
    D_a, _ = _build(spec, class_of=_classes())
    D_b, _ = _build(spec, class_of=1 - _classes())
    np.testing.assert_allclose(D_a, D_b)


# -- orientation ------------------------------------------------------
def test_rows_are_the_receiver():
    """Case 2 is one-sided, so it pins the orientation unambiguously."""
    D = field_matrix(1.0, case=2)
    assert D[0, 1] == -1.0  # a class-A receiver is hostile to a class-B emitter
    assert D[1, 0] == 0.0  # a class-B receiver is indifferent


def test_only_discriminating_receivers_carry_a_field():
    spec = FieldSpec(kind="class", case=6, d=0.7, f_d=0.5)
    D, disc = _build(spec)
    assert np.all(D[~disc] == 0.0)
    assert np.any(D[disc] != 0.0)


def test_discriminating_fraction_matches_f_d():
    spec = FieldSpec(kind="class", case=6, d=0.5, f_d=0.3)
    _, disc = build_field(
        spec, _classes(2000), np.random.default_rng(1), np.random.default_rng(2)
    )
    assert disc.mean() == pytest.approx(0.3, abs=0.03)


def test_zero_d_recovers_the_unperturbed_field_exactly():
    a, _ = _build(FieldSpec(kind="class", case=6, d=0.0, f_d=1.0))
    b, _ = _build(FieldSpec(kind="none", d=0.9, f_d=1.0))
    np.testing.assert_array_equal(a, b)
    assert not a.any()


# -- the controls are matched ----------------------------------------
@pytest.mark.parametrize("kind", ["partition", "pair"])
def test_control_fields_match_the_magnitude_exactly(kind):
    """Same |D| multiset, same perturbed rows: only the alignment differs.

    Without this the control confounds "uncorrelated with class" with "weaker",
    and the paper's central control would prove nothing.
    """
    d, f_d = 0.8, 0.6
    D_cls, disc_cls = _build(FieldSpec(kind="class", case=6, d=d, f_d=f_d))
    D_ctl, disc_ctl = _build(FieldSpec(kind=kind, case=6, d=d, f_d=f_d))

    np.testing.assert_array_equal(disc_cls, disc_ctl)
    np.testing.assert_allclose(
        np.sort(np.abs(D_cls).ravel()), np.sort(np.abs(D_ctl).ravel())
    )


def test_partition_control_is_rank_one_like_the_real_field():
    """It answers "it is just a rank-one perturbation" as well as "just noise"."""
    spec = FieldSpec(kind="partition", case=6, d=0.8, f_d=1.0)
    D, _ = _build(spec)
    assert np.linalg.matrix_rank(D, tol=1e-9) == 1
    D_pair, _ = _build(FieldSpec(kind="pair", case=6, d=0.8, f_d=1.0))
    assert np.linalg.matrix_rank(D_pair, tol=1e-9) > 1


def test_partition_control_is_not_aligned_with_the_true_class():
    """Over many draws the misaligned partition must average to no class signal."""
    cls = _classes()
    kappa = np.where(cls == 0, 1.0, -1.0)
    s = np.outer(kappa, kappa)
    aligns = []
    for seed in range(60):
        D, _ = _build(FieldSpec(kind="partition", case=6, d=1.0, f_d=1.0), seed=seed)
        aligns.append((D * s).mean())
    assert abs(np.mean(aligns)) < 0.1
    # ...while the real field is maximally aligned by construction.
    D_cls, _ = _build(FieldSpec(kind="class", case=6, d=1.0, f_d=1.0))
    assert (D_cls * s).mean() == pytest.approx(1.0, abs=1e-9)


def test_field_kind_does_not_change_who_discriminates():
    """The mask stream is separate from the sign stream, so a control differs
    from its baseline in exactly one respect."""
    masks = []
    for kind in ("class", "partition", "pair"):
        _, disc = _build(FieldSpec(kind=kind, case=6, d=0.5, f_d=0.4))
        masks.append(disc)
    for m in masks[1:]:
        np.testing.assert_array_equal(masks[0], m)


# -- reaching the dynamics -------------------------------------------
def test_zero_field_is_class_blind_in_the_dynamics():
    model = ModelConfig(n_agents=12, n_dim=8, n_issues=3, interactions_per_channel=10)
    key = RunKey("t", "t", point_id({"d": 0.0}), 0, 0)
    b = SocietyBatch.from_keys(
        model, [key], [FieldSpec(kind="class", case=6, d=0.0, f_d=1.0)], master=1
    )
    assert not b.D.any()
