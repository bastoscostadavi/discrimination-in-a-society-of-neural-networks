"""Make the package importable from the tests and register markers.

Its presence here also puts this directory on ``sys.path`` for pytest, so
``pytest`` works from either this directory or the repository root.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: exercises the full dynamics; deselect with -m 'not slow'"
    )
