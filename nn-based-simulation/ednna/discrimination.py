"""Discrimination fields: how class membership perturbs the opinion field.

A prejudiced receiver extends the representation of an issue with
information that is irrelevant to the classification task but correlated with
the *class* of the emitter.  In the algorithm this appears as an additive shift
of the opinion field before the modulation functions are evaluated:

    h_w^D = h_w + D[class(receiver), class(emitter)]

The six cases below are the six matrices of the source draft's Table I: they
differ in whether in-group favouritism, out-group hostility, or both are
present, and in whether one or both classes discriminate.

Sign convention
---------------
`d > 0` means **intolerance toward the out-group and tolerance toward the
in-group**.  Concretely, for a receiver R and emitter E:

    D[R, E] = +d   ->  R is more tolerant of E     (in-group favouritism)
    D[R, E] = -d   ->  R is less tolerant of E     (out-group hostility)

This follows from the algorithm itself.  `F_mu = (1 - 2 Phi(h_w)) g(h_mu)/Z`
is negative when `h_w > 0`, i.e. perceived agreement *builds* trust.  Shifting
`h_w` up therefore makes the receiver read the message as more agreeable and
grow more trusting; shifting it down manufactures perceived disagreement and
breeds distrust.  So the tolerant entry carries `+d`, and with case 6 a society
with `d > 0` ends up trusting in-group and distrusting out-group agents, i.e.
in the discriminatory phase.

The source draft's Table I assigns `-d` to the in-group entries while its text
and figures require `d > 0` to be the discriminatory regime.  Those two cannot
both hold: the draft's table and its Eq. 25, taken literally, put the
discriminatory phase at `d < 0`.  We use the consistent convention above, which
matches the draft's figures and narrative.  `docs/prejudice-field-sign.md`
works the discrepancy through in full and shows both versions side by side; the
`literal_draft=True` flag here reproduces the draft's literal table for that
comparison.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CASES", "field_matrix", "N_CASES"]

#: Index of each class in the discrimination matrix.
CLASS_A, CLASS_B = 0, 1

N_CASES = 6

# Unit templates for the six cases, in the corrected convention where +1 is the
# tolerant entry and -1 the intolerant one.  Rows index the receiver's class,
# columns the emitter's class; entry [R, E] shifts the field of a discriminating
# receiver of class R when it hears from an emitter of class E.
#
#   case 1: A favours its own; B is indifferent.
#   case 2: A is hostile to B; no in-group favouritism; B indifferent.
#   case 3: A both favours its own and is hostile to B; B indifferent.
#   case 4: both classes favour their own, neither is hostile.
#   case 5: both classes are hostile to the other, neither favours its own.
#   case 6: both classes favour their own and are hostile to the other.
_TEMPLATES = {
    1: [[+1.0, 0.0], [0.0, 0.0]],
    2: [[0.0, -1.0], [0.0, 0.0]],
    3: [[+1.0, -1.0], [0.0, 0.0]],
    4: [[+1.0, 0.0], [0.0, +1.0]],
    5: [[0.0, -1.0], [-1.0, 0.0]],
    6: [[+1.0, -1.0], [-1.0, +1.0]],
}

# The four orthogonal components of a class-dependent shift.  Writing
# kappa = +1 for class A and -1 for class B, any 2x2 field decomposes uniquely as
#
#     D[R, E] = a + b*kappa_R + c*kappa_E + p*kappa_R*kappa_E
#
# with one meaning each: `a` a uniform credulity that names no class, `b` a class
# that is more credulous whoever speaks, `c` a class that is believed more whoever
# listens, `p` a dependence on whether the two classes match.  Case 6 is pure `p`.
# The main text varies `p` with a = b = c = 0; the others are here so the same
# sweep machinery can run them.
_KAPPA = (+1.0, -1.0)  # index 0 -> class A, index 1 -> class B
_COMPONENTS = {
    "a": [[1.0, 1.0], [1.0, 1.0]],
    "b": [[kr, kr] for kr in _KAPPA],
    "c": [[ke for ke in _KAPPA] for _ in _KAPPA],
    "p": [[kr * ke for ke in _KAPPA] for kr in _KAPPA],
}
_TEMPLATES.update(_COMPONENTS)

CASES = tuple(sorted(_TEMPLATES, key=str))


def field_matrix(d, case=6, literal_draft=False):
    """The 2x2 prejudice field matrix ``D`` for one case.

    Parameters
    ----------
    d
        Strength of the prejudice field.  Positive `d` is in-group
        tolerance / out-group intolerance; negative `d` is the reverse
        ("reverse discrimination"), which the draft also considers.
    case
        Which of the six cases of Table I to use (1-6).  Case 6, the fully
        symmetric one where both classes favour their own and are hostile to
        the other, is the default and the one the phase diagrams use.
    literal_draft
        If True, flip the overall sign so as to match the source draft's
        Table I read literally together with its Eq. 25.  This puts the
        discriminatory phase at `d < 0` and exists only for the comparison in
        ``docs/prejudice-field-sign.md``.

    Returns
    -------
    (2, 2) array indexed ``[receiver_class, emitter_class]``.
    """
    if case not in _TEMPLATES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    D = np.asarray(_TEMPLATES[case], dtype=float) * float(d)
    return -D if literal_draft else D
