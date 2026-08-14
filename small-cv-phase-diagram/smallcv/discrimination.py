"""Discrimination matrices.

Rows are receiver classes, columns are emitter classes.  A positive entry shifts
the receiver's opinion field upward before evaluating the modulation functions.
"""

from __future__ import annotations

import numpy as np

CASES = (1, 2, 3, 4, 5, 6)
TEMPLATES = {
    1: [[+1.0, 0.0], [0.0, 0.0]],
    2: [[0.0, -1.0], [0.0, 0.0]],
    3: [[+1.0, -1.0], [0.0, 0.0]],
    4: [[+1.0, 0.0], [0.0, +1.0]],
    5: [[0.0, -1.0], [-1.0, 0.0]],
    6: [[+1.0, -1.0], [-1.0, +1.0]],
}


def field_matrix(d, case=6, literal_draft=False):
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    D = np.asarray(TEMPLATES[case], dtype=float) * float(d)
    return -D if literal_draft else D
