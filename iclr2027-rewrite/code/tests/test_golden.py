"""Regression against the reference implementation.

This package was seeded from an existing, tested implementation and then
substantially rewritten: the random streams, the batching, the persistence and
the observable layer are all new.  The rewrite is only safe if the *physics* is
demonstrably unchanged, and "the tests still pass" does not establish that,
because the tests came along with the code.

So these tests drive both implementations from identical injected state through
an identical interaction sequence and require the trajectories to agree to
machine precision.  Random streams differ by design, so a bitwise-identical
*run* is impossible and would not be the right thing to check; what must be
identical is the update itself.
"""

import numpy as np
import pytest

from socsim import FieldSpec, ModelConfig, SocietyBatch
from socsim.observables import correlations, overlaps, trust, _class_relation
from socsim.seeds import RunKey, point_id

pytestmark = pytest.mark.golden

N, K, P = 10, 8, 4


def _pair(reference, d=0.7, f_d=0.6, seed=3):
    """One society in each implementation, sharing state exactly."""
    ref = reference["society"].SocietyBatch(
        n_agents=N, n_dim=K, n_issues=P, d=d, f_d=f_d, case=6, seed=seed
    )
    model = ModelConfig(n_agents=N, n_dim=K, n_issues=P)
    key = RunKey("golden", "golden", point_id({"d": d, "f_d": f_d}), 0, 0)
    new = SocietyBatch.from_keys(
        model, [key], [FieldSpec(kind="class", case=6, d=d, f_d=f_d)], master=1
    )
    # Inject the reference's state, including which agents discriminate and the
    # field they carry, so the only thing left that could differ is the update.
    new.w[:, 0, :] = ref.w[:, 0, :]
    new.C[:, 0] = ref.C[:, 0]
    new.mu[:, :, 0] = ref.mu[:, :, 0]
    new.V[:, :, 0] = ref.V[:, :, 0]
    new.X[:, 0, :] = ref.X[:, 0, :]
    new.D[:, :, 0] = ref.D[:, :, 0]
    new.class_of[:, 0] = ref.class_of
    new.kappa[:, 0] = ref.kappa
    new.discriminates[:, 0] = ref.discriminates[:, 0]
    return ref, new


def test_single_interaction_is_identical(reference):
    ref, new = _pair(reference)
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = int(rng.integers(N))
        e = int(rng.integers(N - 1))
        e += e >= r
        p = int(rng.integers(P))
        ref._interact(r, e, ref.X[p])
        new._interact(r, e, new.X[p])

    np.testing.assert_allclose(new.w[:, 0, :], ref.w[:, 0, :], rtol=0, atol=0)
    np.testing.assert_allclose(new.C[:, 0], ref.C[:, 0], rtol=0, atol=0)
    np.testing.assert_allclose(new.mu[:, :, 0], ref.mu[:, :, 0], rtol=0, atol=0)
    np.testing.assert_allclose(new.V[:, :, 0], ref.V[:, :, 0], rtol=0, atol=0)


@pytest.mark.parametrize("d", [-0.9, 0.0, 0.5])
def test_long_run_is_identical(reference, d):
    """Errors that only appear after annealing would be missed by a short run."""
    ref, new = _pair(reference, d=d)
    rng = np.random.default_rng(11)
    n = 4000
    recv = rng.integers(N, size=n)
    emit = rng.integers(N - 1, size=n)
    emit += emit >= recv
    issue = rng.integers(P, size=n)
    for t in range(n):
        ref._interact(int(recv[t]), int(emit[t]), ref.X[issue[t]])
        new._interact(int(recv[t]), int(emit[t]), new.X[issue[t]])

    np.testing.assert_allclose(new.w[:, 0, :], ref.w[:, 0, :], rtol=0, atol=0)
    np.testing.assert_allclose(new.mu[:, :, 0], ref.mu[:, :, 0], rtol=0, atol=0)
    assert new.n_psd_clips == ref.n_psd_clips


def test_renamed_correlations_equal_the_originals(reference):
    """C_CT/C_CO/C_TO are a relabelling of R_muc/R_cw/R_wmu, not a correction.

    The plan document reads as though the order parameters were being repaired.
    They were not: given the normalisations already in use the two sets are the
    same numbers.  The paper has to say so, or a reader comparing against
    earlier figures will conclude the results moved.  This test is the evidence.
    """
    ref, new = _pair(reference, d=0.8, f_d=0.9)
    rng = np.random.default_rng(5)
    n = 3000
    recv = rng.integers(N, size=n)
    emit = rng.integers(N - 1, size=n)
    emit += emit >= recv
    issue = rng.integers(P, size=n)
    for t in range(n):
        ref._interact(int(recv[t]), int(emit[t]), ref.X[issue[t]])
        new._interact(int(recv[t]), int(emit[t]), new.X[issue[t]])

    old = reference["order_params"].measure(ref)
    q, t_, s = overlaps(new), trust(new), _class_relation(new.kappa)
    got = correlations(q, t_, s)

    assert got["C_CT"][0] == pytest.approx(old["R_muc"][0], abs=1e-12)
    assert got["C_CO"][0] == pytest.approx(old["R_cw"][0], abs=1e-12)
    assert got["C_TO"][0] == pytest.approx(old["R_wmu"][0], abs=1e-12)


def test_balance_matches_the_reference(reference):
    from socsim.observables import balance

    ref, new = _pair(reference, d=0.4, f_d=0.5)
    rng = np.random.default_rng(7)
    for _ in range(2000):
        r = int(rng.integers(N))
        e = int(rng.integers(N - 1))
        e += e >= r
        p = int(rng.integers(P))
        ref._interact(r, e, ref.X[p])
        new._interact(r, e, new.X[p])

    old = reference["order_params"].balance(ref)
    got = balance(overlaps(new), trust(new))
    assert got["B_O"][0] == pytest.approx(old["B_I"][0], abs=1e-12)
    assert got["B_T"][0] == pytest.approx(old["B_A"][0], abs=1e-12)
