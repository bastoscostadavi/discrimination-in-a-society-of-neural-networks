"""The experiment campaign: every run, its role, and what it costs.

Runs are declared here rather than assembled ad hoc in scripts, so that the
whole compute plan is inspectable in one place and the coupling that makes the
controls *paired* with their baseline cannot be broken by accident.

Two invariants hold across the table and both are load-bearing:

* **Controls share ``crn_group`` with their baseline.**  Each control society
  then starts from the same weights, the same distrust and the same interaction
  order as a baseline society at the identical point, so differences can be
  reported paired.  Breaking this silently inflates every control's error bar.
* **Control grids are stride-subgrids of the baseline grid.**  A coarser grid
  built with its own ``linspace`` would land on different points and destroy the
  pairing, which is why :meth:`GridSpec.subgrid` exists and why the point counts
  are odd.

Measured throughput on this machine is about 3.0 societies/second at ``N = 40``,
``K = 30``, 500 interactions per ordered pair, with ten workers saturated ---
roughly 10,800 societies per hour, scaling as ``N^2`` through the interaction
count.
"""

from __future__ import annotations

from .config import GridSpec, ModelConfig, RunSpec

__all__ = ["CAMPAIGN", "get_run", "cost_table", "LOAD_BEARING", "CUT_ORDER"]

_BASE = ModelConfig(n_agents=40, n_dim=30, n_issues=5)
_FINE = GridSpec(n_d=49, n_fd=49, n_init=24)

#: Coarser grids, each an exact subgrid of ``_FINE``.
_MID = _FINE.subgrid(2).with_(n_init=16)  # 25 x 25
_COARSE = _FINE.subgrid(3).with_(n_init=16)  # 17 x 17
_FAST = _FINE.subgrid(4).with_(n_init=12)  # 13 x 13

CAMPAIGN = {
    # -- gates: cheap, and everything downstream depends on them ------
    "B1_null": RunSpec(
        name="B1_null",
        model=_BASE,
        grid=GridSpec(n_d=3, n_fd=3, n_init=200),
        crn_group="main_P5",
        n_permutations=400,
    ),
    "H1_schedule_audit": RunSpec(
        name="H1_schedule_audit",
        model=_BASE,
        grid=GridSpec(n_d=3, n_fd=3, n_init=40),
        crn_group="audit",
    ),
    # -- the main result ----------------------------------------------
    "A1_main_P5": RunSpec(
        name="A1_main_P5", model=_BASE, grid=_FINE, crn_group="main_P5"
    ),
    "A2_main_P100": RunSpec(
        name="A2_main_P100",
        model=_BASE.with_(n_issues=100),
        grid=_MID,
        crn_group="main_P100",
    ),
    # -- controls, all paired against A1 ------------------------------
    "C2a_partition": RunSpec(
        name="C2a_partition",
        model=_BASE,
        grid=_COARSE,
        field_kind="partition",
        crn_group="main_P5",
    ),
    "C2b_pair": RunSpec(
        name="C2b_pair",
        model=_BASE,
        grid=_FAST,
        field_kind="pair",
        crn_group="main_P5",
    ),
    "C4_frozen_trust": RunSpec(
        name="C4_frozen_trust",
        model=_BASE.with_(freeze=("trust",)),
        grid=_COARSE,
        crn_group="main_P5",
    ),
    "C5_frozen_opinion": RunSpec(
        name="C5_frozen_opinion",
        model=_BASE.with_(freeze=("opinion",)),
        grid=_COARSE,
        crn_group="main_P5",
    ),
    # -- finite size: where a real threshold might live ---------------
    "D1_N20": RunSpec(
        name="D1_N20", model=_BASE.with_(n_agents=20), grid=_COARSE, crn_group="fs_N20"
    ),
    "D2_N60": RunSpec(
        name="D2_N60",
        model=_BASE.with_(n_agents=60),
        grid=_COARSE.with_(n_init=16),
        crn_group="fs_N60",
    ),
    "D3_N80": RunSpec(
        name="D3_N80",
        model=_BASE.with_(n_agents=80),
        grid=_FAST,
        crn_group="fs_N80",
    ),
    # -- appendix ------------------------------------------------------
    **{
        f"F1_case{c}": RunSpec(
            name=f"F1_case{c}",
            model=_BASE,
            grid=_FAST.with_(n_init=8),
            case=c,
            crn_group="main_P5",
        )
        for c in (1, 2, 3, 4, 5)
    },
    "G1_replica": RunSpec(
        name="G1_replica",
        model=_BASE,
        grid=_FAST.with_(n_init=2, n_disorder=12),
        crn_group="replica",
    ),
}

#: Needed for the nine-page main text.  These are never cut.
LOAD_BEARING = (
    "B1_null",
    "H1_schedule_audit",
    "A1_main_P5",
    "A2_main_P100",
    "C2a_partition",
    "D1_N20",
    "D2_N60",
)

#: If the budget runs short, drop from the front of this list.
CUT_ORDER = (
    "F1_case1",
    "F1_case2",
    "F1_case3",
    "F1_case4",
    "F1_case5",
    "C2b_pair",
    "G1_replica",
    "D3_N80",
    "C5_frozen_opinion",
    "C4_frozen_trust",
)


def get_run(name):
    if name not in CAMPAIGN:
        raise ValueError(f"unknown run {name!r}; choose from {sorted(CAMPAIGN)}")
    return CAMPAIGN[name]


def cost_table(rate_at_N40=3.0):
    """Estimated wall-clock per run, at the measured throughput.

    Cost scales with the interaction count, which is ``Delta t * N (N-1)``, so
    the anchor measured at ``N = 40`` is rescaled by ``N^2``.
    """
    rows = []
    for name, spec in CAMPAIGN.items():
        n = spec.grid.n_societies
        N = spec.model.n_agents
        rate = rate_at_N40 * (40 * 39) / (N * (N - 1))
        rate *= 500.0 / spec.model.interactions_per_channel
        if "opinion" in spec.model.freeze:
            rate *= 8.0  # no covariance update to carry
        hours = n / rate / 3600.0
        rows.append(
            {
                "run": name,
                "N": N,
                "grid": f"{spec.grid.n_d}x{spec.grid.n_fd}",
                "reps": spec.grid.n_init * spec.grid.n_disorder,
                "societies": n,
                "hours": hours,
                "load_bearing": name in LOAD_BEARING,
            }
        )
    return rows
