"""The four-component basis: orthogonality, round-trip, and the tabulated cases."""

import numpy as np
import pytest

from credfield.fields import (
    COMPONENTS,
    KAPPA,
    TABLE_I,
    WEIGHTS,
    decompose,
    field_matrix,
    pure,
)


def test_basis_is_orthogonal_with_squared_norm_four():
    """What makes :func:`decompose` a projection divided by four."""
    for i, ki in enumerate(COMPONENTS):
        for kj in COMPONENTS[i:]:
            got = float(np.sum(WEIGHTS[ki] * WEIGHTS[kj]))
            assert got == pytest.approx(4.0 if ki == kj else 0.0, abs=1e-12)


@pytest.mark.parametrize("q", [(0.1, 0.2, 0.3, 0.4), (0.0, -1.0, 0.5, 0.0),
                               (-0.25, 0.0, 0.0, 1.0)])
def test_decompose_inverts_field_matrix(q):
    np.testing.assert_allclose(decompose(field_matrix(*q)), q, atol=1e-12)


def test_every_two_by_two_matrix_is_reachable():
    """Four components, four entries: the basis spans, so nothing is unrepresentable."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        M = rng.normal(size=(2, 2))
        np.testing.assert_allclose(field_matrix(*decompose(M)), M, atol=1e-12)


def test_pure_components_have_the_advertised_structure():
    """Each component's signature, read straight off the matrix."""
    # a: the same entry everywhere, referring to no label
    assert np.allclose(pure("a", 0.7), 0.7)
    # b: rows differ, columns do not -- depends on the listener alone
    b = pure("b", 0.7)
    assert np.allclose(b[0], +0.7) and np.allclose(b[1], -0.7)
    assert np.allclose(b[:, 0], b[:, 1])
    # c: columns differ, rows do not -- depends on the speaker alone
    c = pure("c", 0.7)
    assert np.allclose(c[:, 0], +0.7) and np.allclose(c[:, 1], -0.7)
    assert np.allclose(c[0], c[1])
    # p: depends on whether the two match
    p = pure("p", 0.7)
    assert p[0, 0] == p[1, 1] == pytest.approx(+0.7)
    assert p[0, 1] == p[1, 0] == pytest.approx(-0.7)


def test_relabelling_the_classes_flips_b_and_c_only():
    """The symmetry that makes b and c name a class rather than a relation."""
    a, b, c, p = 0.3, -0.4, 0.5, 0.6
    D = field_matrix(a, b, c, p)
    swapped = D[::-1, ::-1]  # exchange the roles of A and B
    np.testing.assert_allclose(decompose(swapped), (a, -b, -c, p), atol=1e-12)


def test_kappa_is_the_class_variable_of_the_paper():
    np.testing.assert_allclose(KAPPA, [+1.0, -1.0])


def test_table_one_cases_decompose_as_documented():
    expected = {
        1: (0.25, 0.25, 0.25, 0.25),
        2: (-0.25, -0.25, 0.25, 0.25),
        3: (0.0, 0.0, 0.5, 0.5),
        4: (0.5, 0.0, 0.0, 0.5),
        5: (-0.5, 0.0, 0.0, 0.5),
        6: (0.0, 0.0, 0.0, 1.0),
    }
    for case, q in expected.items():
        np.testing.assert_allclose(TABLE_I[case], q, atol=1e-12,
                                   err_msg=f"case {case}")


def test_only_case_six_is_a_pure_component():
    """The reason a status asymmetry has not been looked at on its own."""
    pure_cases = [case for case, q in TABLE_I.items()
                  if sum(abs(v) > 1e-12 for v in q) == 1]
    assert pure_cases == [6]


def test_status_never_appears_alone_in_the_table():
    """Wherever c is present it is matched by an equal p, so it is never isolated."""
    for case, (a, b, c, p) in TABLE_I.items():
        if abs(c) > 1e-12:
            assert c == pytest.approx(p), f"case {case}: c={c}, p={p}"


def test_pure_p_reproduces_the_main_line_of_work_case_six():
    """The bridge to ``ednna.discrimination``: same matrix, so same dynamics."""
    template = np.array([[+1.0, -1.0], [-1.0, +1.0]])
    np.testing.assert_allclose(pure("p", 0.42), 0.42 * template, atol=1e-12)


def test_field_matrix_broadcasts_over_a_grid_axis():
    strengths = np.linspace(-1, 1, 7)
    D = field_matrix(c=strengths)
    assert D.shape == (7, 2, 2)
    for i, s in enumerate(strengths):
        np.testing.assert_allclose(D[i], pure("c", s), atol=1e-12)


def test_pure_rejects_an_unknown_component():
    with pytest.raises(ValueError, match="component must be one of"):
        pure("d", 1.0)


def test_decompose_rejects_the_wrong_shape():
    with pytest.raises(ValueError, match=r"\(\.\.\., 2, 2\)"):
        decompose(np.zeros((3, 3)))
