"""Path setup shared by the scripts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIBLING = ROOT.parent / "nn-based-simulation"

for p in (ROOT, SIBLING):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def theory():
    """``(F_w, F_mu)`` from the simulation package, or ``(None, None)``.

    Imported rather than transcribed: two copies of the paper's equations in one
    repository is one copy too many.
    """
    try:
        from ednna.modulation import F_mu, F_w
        return F_w, F_mu
    except ImportError:
        return None, None
