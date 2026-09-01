"""Import paths for pytest: this package, and the simulation package next door.

The theory is imported from ``nn-based-simulation`` rather than transcribed, so
the tests that check a sign convention are checking it against the same code the
paper's figures are drawn from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT.parent / "nn-based-simulation"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
