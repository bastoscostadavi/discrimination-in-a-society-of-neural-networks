"""The discrimination field, and the controls that isolate what it does.

A discriminating receiver extends its representation of an incoming *message*
with one extra coordinate that carries no information about the issue and
depends only on the source's class.  In the algorithm this is an additive shift
of the agreement field before the modulation functions are evaluated:

    h_w^D = h_w + D[class(receiver), class(emitter)]

Sign convention
---------------
``d > 0`` means **tolerance towards the in-group and intolerance towards the
out-group**.  This is forced by the algorithm, not chosen.  ``F_mu`` carries the
prefactor ``(1 - 2 Phi(h_w))``, which is negative when ``h_w > 0``, and ``mu`` is
*distrust*; so perceived agreement drives ``mu`` down and builds trust.  Adding
``+d`` therefore makes a receiver read its counterpart as more agreeable than it
is and grow more trusting --- the tolerant response --- and ``-d`` manufactures
disagreement and breeds distrust.  Hence for case 6:

    D = d * [[+1, -1],
             [-1, +1]]              rows: receiver's class, cols: emitter's

with the *tolerant* entry on the diagonal.  Compactly, ``D_{e|r} = z_r d k_r k_e``
with ``k = +-1`` the signed class label and ``z_r`` the discriminator indicator.

Note the opposite sign appears in the literature this model comes from, where a
table assigning ``-d`` to the in-group coexists with text and figures placing the
discriminatory regime at ``d > 0``.  Those two cannot both hold: taken together
they put the discriminatory regime at ``d < 0`` and mirror every map in ``d``.
``convention="legacy"`` reproduces that reading for the side-by-side comparison
in the appendix; nothing else uses it.

The control fields
------------------
Three of the paper's controls are alternative fields of *identical magnitude*,
differing only in what the sign is correlated with.  Keeping them here, rather
than as forks of the dynamics, is what makes "same perturbation, different
alignment" a checkable property instead of a claim --- see
``tests/test_discrimination.py``, which asserts the multiset of ``|D|`` entries is
identical across all of them.

``class``
    The real thing: sign follows the true class relation.
``partition``
    Sign follows a random balanced partition drawn *independently of the true
    class*.  Same magnitude, same rank-one structure, same entry distribution;
    the only difference is which partition the field aligns to.  This is the
    sharpest control, because it answers "it is just a rank-one perturbation"
    and "it is just noise" at the same time.
``pair``
    Sign drawn i.i.d. per ordered pair and quenched.  Full-rank noise.
``none``
    No field at all; must be bit-identical to ``class`` with ``d = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "CASES",
    "N_CASES",
    "FIELD_KINDS",
    "FieldSpec",
    "field_matrix",
    "build_field",
]

#: Index of each class in the discrimination matrix.
CLASS_A, CLASS_B = 0, 1

N_CASES = 6
FIELD_KINDS = ("class", "partition", "pair", "none")

# Unit templates for the six inequivalent fillings, with +1 the tolerant entry
# and -1 the intolerant one.  Rows index the receiver's class, columns the
# emitter's: entry [R, E] shifts the field of a discriminating receiver of class
# R when it hears from an emitter of class E.
#
#   case 1: A favours its own; B is indifferent.
#   case 2: A is hostile to B; no in-group favouritism; B indifferent.
#   case 3: A both favours its own and is hostile to B; B indifferent.
#   case 4: both classes favour their own, neither is hostile.
#   case 5: both classes are hostile to the other, neither favours its own.
#   case 6: both favour their own and are hostile to the other.  The symmetric
#           case, in which the two classes play equivalent roles, so any
#           asymmetry in the results is spontaneous.  All main-text runs use it.
_TEMPLATES = {
    1: [[+1.0, 0.0], [0.0, 0.0]],
    2: [[0.0, -1.0], [0.0, 0.0]],
    3: [[+1.0, -1.0], [0.0, 0.0]],
    4: [[+1.0, 0.0], [0.0, +1.0]],
    5: [[0.0, -1.0], [-1.0, 0.0]],
    6: [[+1.0, -1.0], [-1.0, +1.0]],
}

CASES = tuple(sorted(_TEMPLATES))


@dataclass(frozen=True)
class FieldSpec:
    """Which perturbation a run applies, and how strong it is.

    ``kind`` selects the alignment (see the module docstring); ``case`` selects
    the template and is meaningful only for ``kind="class"`` and
    ``kind="partition"``.
    """

    kind: str = "class"
    case: int = 6
    d: float = 0.0
    f_d: float = 0.0
    convention: str = "algorithmic"  # or "legacy", for the appendix comparison

    def __post_init__(self):
        if self.kind not in FIELD_KINDS:
            raise ValueError(f"kind must be one of {FIELD_KINDS}, got {self.kind!r}")
        if self.case not in _TEMPLATES:
            raise ValueError(f"case must be one of {CASES}, got {self.case!r}")
        if self.convention not in ("algorithmic", "legacy"):
            raise ValueError("convention must be 'algorithmic' or 'legacy'")
        if not 0.0 <= self.f_d <= 1.0:
            raise ValueError(f"f_d must lie in [0, 1], got {self.f_d!r}")

    def with_(self, **kw):
        return replace(self, **kw)


def field_matrix(d, case=6, convention="algorithmic"):
    """The 2x2 field matrix ``D`` indexed ``[receiver_class, emitter_class]``.

    ``d > 0`` is in-group tolerance and out-group intolerance; ``d < 0`` is the
    reverse, which the phase diagram also covers.  ``convention="legacy"``
    negates the matrix, reproducing the inconsistent reading described in the
    module docstring.
    """
    if case not in _TEMPLATES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    D = np.asarray(_TEMPLATES[case], dtype=float) * float(d)
    return -D if convention == "legacy" else D


def build_field(spec, class_of, rng_mask, rng_field, dtype=np.float64):
    """The per-agent field of one society.

    Parameters
    ----------
    spec
        Which perturbation to apply.
    class_of
        ``(N,)`` array of 0/1 true class labels.
    rng_mask, rng_field
        Separate generators for *who* discriminates and for *what sign* they
        apply.  Keeping them apart means the discriminator mask is identical
        across ``kind`` variants at the same seed, so a control differs from the
        baseline in exactly one respect.

    Returns
    -------
    ``(D, discriminates)`` with ``D`` of shape ``(N, N)`` indexed
    ``[receiver, emitter]`` and ``discriminates`` of shape ``(N,)``.

    Every row of a non-discriminating receiver is exactly zero, so the fraction
    of the population that is perturbed --- and the magnitude it is perturbed by
    --- is the same for every ``kind``.
    """
    N = class_of.size
    discriminates = rng_mask.random(N) < spec.f_d

    if spec.kind == "none" or spec.d == 0.0:
        return np.zeros((N, N), dtype=dtype), discriminates

    if spec.kind == "class":
        M = field_matrix(spec.d, spec.case, spec.convention)
        D = M[np.ix_(class_of, class_of)]

    elif spec.kind == "partition":
        # A balanced partition drawn independently of the true class: same
        # magnitude and same rank-one structure, aligned to the wrong axis.
        g = np.zeros(N, dtype=np.int8)
        g[rng_field.permutation(N)[: N // 2]] = 1
        M = field_matrix(spec.d, spec.case, spec.convention)
        D = M[np.ix_(g, g)]

    elif spec.kind == "pair":
        # Quenched i.i.d. signs per ordered pair.  Scaled by the mean absolute
        # entry of the corresponding class template so that the magnitude
        # matches even for the templates with zero entries (cases 1-5).
        M = field_matrix(spec.d, spec.case, spec.convention)
        scale = float(np.abs(M).mean())
        signs = np.where(rng_field.random((N, N)) < 0.5, -1.0, 1.0)
        D = signs * scale

    else:  # pragma: no cover - guarded by FieldSpec.__post_init__
        raise ValueError(f"unknown field kind {spec.kind!r}")

    D = np.asarray(D, dtype=dtype) * discriminates[:, None]
    return np.ascontiguousarray(D), discriminates
