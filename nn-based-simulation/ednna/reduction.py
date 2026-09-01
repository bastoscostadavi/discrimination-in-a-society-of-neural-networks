"""Reducing a microstate to a plane, for looking at a society directly.

The order parameters of :mod:`ednna.order_params` average a society down to five
numbers.  This module does the other thing one can do with a microstate: keep every
agent and throw away dimensions instead, so that a population can be drawn.

One operation serves both sectors.  Each agent contributes one vector --- its
weights ``w_I`` in ``R^K`` for the opinion sector, its outgoing trust profile
``eta_{.|I}`` in ``R^N`` for the trust sector --- the vectors are normalized to unit
length, and the population is projected onto the leading right singular vectors of
the resulting matrix.  Two agents come out close when they hold the same opinion,
or trust the same people, respectively.

Used by ``scripts/state_portraits.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["unit_rows", "project"]


def unit_rows(v):
    """Rows scaled to unit length, safe against a zero row."""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def project(vectors, n_comp, reference=None, positive_class=None):
    """Project unit-normalized rows onto their own leading axes.

    Parameters
    ----------
    vectors
        ``(N, D)``: one vector per agent, in either sector's space.
    n_comp
        How many components to keep.
    reference
        Optional vector in the same ``R^D`` as the rows, projected alongside them
        and used to fix the frame; see the gauge discussion below.
    positive_class
        Optional boolean mask over agents, used for the weaker gauge available when
        there is no reference.

    Returns
    -------
    coords : ``(N, n_comp)``
        The agents' coordinates.
    captured : float
        Fraction of the total squared row norm the retained components carry.  Since
        every row is a unit vector this is also the mean squared length of the
        coordinates, so it says how much of a typical agent's vector the picture
        shows: near one when the sector has collapsed onto few directions, small when
        it has no low-dimensional structure at all.
    ref : ``(n_comp,)`` or None
        ``reference`` in the same coordinates.

    Two properties of the decomposition matter for reading the result.

    It is **uncentered**: the origin is the zero vector rather than the population
    mean, so two opposed camps stay opposed instead of collapsing into one cloud
    split down the middle.

    Its axes are fixed only **up to sign and an in-plane rotation**, and a picture
    read geometrically needs a gauge for both.  A ``reference`` gives an absolute
    one: the frame is rotated so the reference lies along ``+x``, which is what lets
    two panels be compared to each other and makes the sign of the alignment
    between the agents and the reference visible.  In the trust sector the reference
    is the class indicator, and that sign is exactly the difference between a
    population that trusts its own class and one that trusts the other.  Failing a
    reference, only the sign of the first axis can be fixed, by asking that
    ``positive_class`` have a non-negative mean coordinate along it.  Whatever
    remains free is fixed by asking that the agent furthest out along each axis have
    a positive coordinate.
    """
    unit = unit_rows(np.asarray(vectors, dtype=float))
    _, s, vt = np.linalg.svd(unit, full_matrices=False)
    axes = vt[:n_comp]
    coords = unit @ axes.T
    ref = None
    if reference is not None:
        ref = unit_rows(np.asarray(reference, dtype=float)) @ axes.T

    free = list(range(n_comp))
    if ref is not None and np.linalg.norm(ref) > 1e-9:
        # An orthonormal frame whose first vector is the reference.  Rotating the
        # coordinates into it moves no agent relative to any other.
        basis = np.linalg.qr(np.column_stack([ref, np.eye(n_comp)]))[0][:, :n_comp]
        basis = basis * (np.sign(basis[:, 0] @ ref) or 1.0)
        coords, ref, free = coords @ basis, ref @ basis, free[1:]
    elif positive_class is not None and np.any(positive_class):
        flip = np.sign(coords[positive_class, 0].mean()) or 1.0
        coords[:, 0] *= flip
        if ref is not None:
            ref[0] *= flip
        free = free[1:]
    for c in free:
        coords[:, c] *= np.sign(coords[np.argmax(np.abs(coords[:, c])), c]) or 1.0

    captured = float((s[:n_comp] ** 2).sum() / (s**2).sum())
    return coords, captured, ref
