"""Deriving every random stream from a key, and why the split matters.

The reference implementation seeded one generator per *batch*
(``seed = base + flat_offset``), so the randomness a grid point received
depended on the batch size, on the grid resolution, and on where the point
happened to fall in the flattened array.  Nothing was reproducible under
re-gridding, a single society could not be re-run in isolation to debug it, and
adding points by adaptive refinement perturbed the points already computed.

Here every society is identified by a :class:`RunKey`, and every random stream
it needs is derived from that key by hashing.  Three properties follow, and each
is a test in ``tests/test_seeds.py``:

**Batch invariance.**  A society's trajectory depends only on its key --- never
on chunk size, worker count, completion order, or grid resolution.  Adaptive
refinement appends new points without disturbing old ones, and an interrupted
campaign resumes by set difference.

**Separated disorder.**  The key splits the randomness in two.  ``disorder``
indexes the quenched environment (the agenda, the class labels, which agents
discriminate, the field's signs); ``init`` indexes the initial state and the
interaction schedule.  Holding ``disorder`` fixed while varying ``init`` gives
independent replicas in the *same* environment, which is exactly the replica
overlap needed to decide whether the frustrated region deserves to be called a
glass or merely disordered.

**Common random numbers, deliberately.**  The ``init`` and ``schedule`` streams
key off ``crn_group`` rather than ``experiment``.  Give a control the same
``crn_group`` and the same grid as its baseline and each control society is
*paired* with a baseline society sharing initial weights, initial distrust and
interaction order.  Differences can then be reported paired, which removes the
between-society variance that would otherwise dominate.  This costs nothing and
is the single largest variance reduction available here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

__all__ = ["RunKey", "point_id", "stream", "ROLES", "DISORDER_ROLES", "INIT_ROLES"]

#: Roles whose stream is indexed by ``disorder`` -- the quenched environment.
DISORDER_ROLES = ("agenda", "mask", "field", "classes")

#: Roles whose stream is indexed by ``init`` -- the state and the schedule.
#: These key off ``crn_group``, so they are shared between paired runs.
INIT_ROLES = ("init", "schedule")

ROLES = DISORDER_ROLES + INIT_ROLES


def _digest(*parts):
    payload = "|".join(str(p) for p in parts).encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def point_id(params):
    """A stable short identifier for a point in parameter space.

    Hashes the *values*, not a grid index, so the same ``(d, f_d)`` appearing in
    two experiments or at two grid resolutions gets the same identifier and the
    two runs are directly comparable.  Floats are rounded to 12 decimal places
    first, so that ``np.linspace`` round-off cannot split one point into two.
    """
    canon = {}
    for k, v in sorted(params.items()):
        if isinstance(v, (float, np.floating)):
            v = round(float(v) + 0.0, 12)  # +0.0 normalises -0.0 to 0.0
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        canon[k] = v
    return hashlib.blake2b(
        json.dumps(canon, sort_keys=True, default=str).encode(), digest_size=8
    ).hexdigest()


@dataclass(frozen=True)
class RunKey:
    """Everything that identifies one society.

    ``experiment`` separates campaigns; ``crn_group`` is what a baseline and its
    controls share so their initial conditions coincide; ``point_id`` locates
    the point in parameter space; ``disorder`` and ``init`` index the two
    independent families of replicates.
    """

    experiment: str
    crn_group: str
    point_id: str
    disorder: int = 0
    init: int = 0

    def as_row(self):
        return {
            "experiment": self.experiment,
            "crn_group": self.crn_group,
            "point_id": self.point_id,
            "replicate_dis": int(self.disorder),
            "replicate_init": int(self.init),
        }


def stream(key, role, master):
    """A fresh generator for one ``(key, role)``.

    ``role`` must be one of :data:`ROLES`.  Disorder roles are namespaced by
    ``experiment`` and indexed by ``key.disorder``; init roles are namespaced by
    ``crn_group`` and indexed by ``key.init``, which is what creates the pairing
    between a baseline and its controls.
    """
    return np.random.default_rng(np.random.SeedSequence(derived_seed(key, role, master)))


def derived_seed(key, role, master):
    """The integer seed behind :func:`stream`, recorded in the output file.

    Storing it costs 8 bytes per society and makes any single run
    re-executable straight from the results file.

    ``"schedule"`` is the one role that deliberately **omits** ``point_id``.
    The inner loop gets its speed from advancing many societies in lockstep,
    which requires them to share the sequence of ``(receiver, emitter, issue)``
    draws; making the schedule depend on the point would force per-society
    gathers and cost roughly a factor of three.  Sharing it couples only the
    *order* of interactions across the grid, never the content: initial
    weights, initial distrust, the agenda, the class labels and the
    discriminator mask are all still drawn per society.  Replicates therefore
    still see genuinely different schedules, which is what the uncertainty
    estimate needs; what they share is within a replicate, across pixels, where
    it acts as a variance-reducing common random number rather than as a bias.
    ``tests/test_seeds.py`` pins this, and an audit run compares the estimated
    seed-to-seed spread against fully independent schedules.
    """
    if role == "schedule":
        return int.from_bytes(_digest(master, key.crn_group, role, key.init)[:8], "little")
    if role in INIT_ROLES:
        namespace, index = key.crn_group, key.init
    elif role in DISORDER_ROLES:
        namespace, index = key.experiment, key.disorder
    else:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return int.from_bytes(_digest(master, namespace, key.point_id, role, index)[:8], "little")
