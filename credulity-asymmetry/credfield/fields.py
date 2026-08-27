"""The four components of a class-dependent field shift, and how to build one.

A prejudiced receiver shifts its opinion field by an amount that may depend on
its own class and on the emitter's::

    h_w^D = h_w + D[class(receiver), class(emitter)]

``D`` is a 2x2 matrix, so four numbers are free.  Since each class index takes
two values, writing ``kappa = +1`` for class A and ``-1`` for class B gives an
orthogonal basis for those four numbers::

    D[r, e] = a + b kappa_r + c kappa_e + p kappa_r kappa_e

with one meaning each:

``a`` uniform credulity
    the same shift whoever is speaking and whoever is listening.  Refers to no
    label at all, so it cannot correlate anything with class; it moves the trust
    separatrix to ``h_w = -a`` and so changes how readily trust forms.

``b`` credulity asymmetry
    depends only on who is *listening*.  One class is systematically the more
    credulous; it trusts everyone, the other class distrusts everyone.

``c`` status
    depends only on who is *speaking*.  One class is believed more by everyone,
    its own members included.  This is prestige, or its negative, stigma:
    an advantage attached to a class rather than to a relationship.

``p`` matching
    depends only on whether the two classes agree.  In-group favouritism and
    out-group hostility at once.  This is the component the main line of work
    studies, and the only one that both refers to the label and survives
    relabelling the classes.

Relabelling sends ``(a, b, c, p) -> (a, -b, -c, p)``, so ``b`` and ``c`` name a
class rather than a relation between classes.  That is a reason to be careful
reading their results, not a reason to skip them: a real status asymmetry is
exactly a fact about which class is which.

Where the six tabulated cases sit
---------------------------------
The companion manuscript's Table I lists six fillings of ``D``.  Decomposed in
this basis (:data:`TABLE_I`) they are:

===== ======== ======== ======== ========  ================================
case      a        b        c        p     what it is
===== ======== ======== ======== ========  ================================
1       1/4      1/4      1/4      1/4    A favours its own; B indifferent
2      -1/4     -1/4      1/4      1/4    A hostile to B; B indifferent
3        0        0       1/2      1/2    both of the above
4       1/2       0        0       1/2    both classes favour their own
5      -1/2       0        0       1/2    both hostile to the other
6        0        0        0        1     the symmetric case, pure ``p``
===== ======== ======== ======== ========  ================================

Two things are worth reading off that table.  Only case 6 is a pure component.
And ``c`` appears in exactly the three cases where one class discriminates and
the other does not, always in equal measure with ``p`` and never alone -- which
is presumably why a status asymmetry has not been looked at on its own: no entry
in the table isolates it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COMPONENTS", "KAPPA", "CLASS_A", "CLASS_B", "field_matrix",
           "decompose", "TABLE_I", "WEIGHTS"]

#: Index of each class into a 2x2 field matrix.
CLASS_A, CLASS_B = 0, 1

#: The class variable itself: ``kappa_A = +1``, ``kappa_B = -1``.
KAPPA = np.array([+1.0, -1.0])

#: Component names, in the order used everywhere.
COMPONENTS = ("a", "b", "c", "p")

#: The basis, as four 2x2 weight matrices indexed ``[receiver, emitter]``.  They
#: are mutually orthogonal under the entrywise inner product and each has squared
#: norm 4, which is what makes :func:`decompose` a division by four.
WEIGHTS = {
    "a": np.ones((2, 2)),
    "b": np.repeat(KAPPA[:, None], 2, axis=1),
    "c": np.repeat(KAPPA[None, :], 2, axis=0),
    "p": KAPPA[:, None] * KAPPA[None, :],
}


def field_matrix(a=0.0, b=0.0, c=0.0, p=0.0):
    """The 2x2 field matrix for one set of components.

    Rows index the receiver's class, columns the emitter's, class A first.
    Entry ``[r, e]`` is what a prejudiced receiver of class ``r`` adds to its
    opinion field when it hears from an emitter of class ``e``.

    Sign convention, inherited from the algorithm and unchanged here: a positive
    entry makes the receiver read the message as more agreeable than it is, and
    since perceived agreement is what builds trust, a positive entry is
    *tolerance*.  So ``p > 0`` is in-group tolerance with out-group intolerance,
    and ``c > 0`` means class A is the credited class.

    Scalars give a ``(2, 2)`` array; array arguments broadcast, giving
    ``(..., 2, 2)`` with the class indices last.
    """
    a, b, c, p = (np.asarray(v, dtype=float)[..., None, None]
                  for v in np.broadcast_arrays(a, b, c, p))
    return a * WEIGHTS["a"] + b * WEIGHTS["b"] + c * WEIGHTS["c"] + p * WEIGHTS["p"]


def decompose(D):
    """``(a, b, c, p)`` of a 2x2 field matrix.  Inverse of :func:`field_matrix`.

    The basis is orthogonal with squared norm 4 in each direction, so each
    component is the entrywise inner product with its weight matrix over four.
    """
    D = np.asarray(D, dtype=float)
    if D.shape[-2:] != (2, 2):
        raise ValueError(f"expected a (..., 2, 2) matrix, got shape {D.shape}")
    return tuple(float(np.sum(D * WEIGHTS[k]) / 4.0) if D.ndim == 2
                 else np.sum(D * WEIGHTS[k], axis=(-2, -1)) / 4.0
                 for k in COMPONENTS)


def pure(component, strength=1.0):
    """The field with one component set and the other three zero."""
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    return field_matrix(**{component: strength})


#: The six cases of the companion manuscript's Table I, in this basis.  Built
#: from the same unit templates the main line of work uses, so a change there
#: that contradicted this table would show up as a test failure rather than as a
#: disagreement between two directories.
_TABLE_I_TEMPLATES = {
    1: [[+1.0, 0.0], [0.0, 0.0]],
    2: [[0.0, -1.0], [0.0, 0.0]],
    3: [[+1.0, -1.0], [0.0, 0.0]],
    4: [[+1.0, 0.0], [0.0, +1.0]],
    5: [[0.0, -1.0], [-1.0, 0.0]],
    6: [[+1.0, -1.0], [-1.0, +1.0]],
}

TABLE_I = {case: decompose(np.asarray(M)) for case, M in _TABLE_I_TEMPLATES.items()}
