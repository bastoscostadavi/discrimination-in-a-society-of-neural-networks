"""Test configuration.

The cross-implementation tests compare against the reference implementation in
``nn-based-simulation/``, which is not a package and is not a dependency of this
one.  It is imported here by path so the regression can run while that directory
still exists, and skipped cleanly once it does not --- the anonymous supplement
must stand alone.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
REFERENCE = REPO / "nn-based-simulation"


@pytest.fixture(scope="session")
def reference():
    """The reference package, or a skip if it is not present."""
    if not (REFERENCE / "ednna").is_dir():
        pytest.skip("reference implementation not available")
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    import ednna  # noqa: F401
    import ednna.order_params as op
    import ednna.society as soc

    return {"society": soc, "order_params": op}
