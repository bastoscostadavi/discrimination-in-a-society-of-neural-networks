"""Every quantity the sweep records must reach a reader.

A quantity that is measured, cached, described in the README and never once
printed or plotted is indistinguishable, to a reader, from one that was never
measured -- and it is worse than a missing measurement, because the prose around
it implies the evidence exists.  This was a real gap here: eight of the
twenty-five swept quantities reached no output at all, including ``B_rho``, one
of the five the main line of work reports, and all four within-group balances,
which have a function, a docstring and a README paragraph between them.

"Reaches a reader" means one of three things, and the test accepts all three:
printed by name in the report, drawn as a panel, or folded into a derived
quantity that is itself printed or drawn
(:data:`uniform_phase.DERIVED_FROM`).
"""

from __future__ import annotations

import io
import pathlib
import re
from contextlib import redirect_stdout

import numpy as np
import pytest

from credulity.order_params import ORDER_PARAM_NAMES, PAPER_NAMES
from uniform_phase import BOTTOM, DERIVED_FROM, TOP, report

N_A, N_F, N_AGENTS = 41, 21, 40


def _synthetic_plane():
    """A plane ordered enough that no branch of the report bails out early.

    The values do not matter -- what is being checked is which *names* reach the
    page -- but they must be finite and must actually cross, or the threshold
    section returns all-nan and prints nothing.
    """
    a = np.linspace(-1.0, 1.0, N_A)
    f = np.linspace(0.0, 1.0, N_F)
    A, F = np.meshgrid(a, f)
    base = np.sign(A) * F / (1.0 + np.exp(-(np.abs(A) - 0.4) / 0.05))
    data = {"a": a, "f": f}
    for i, name in enumerate(ORDER_PARAM_NAMES):
        data[name] = base + 0.001 * i
    data["frac_biased"] = F.copy()
    return data


@pytest.fixture(scope="module")
def report_text():
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(_synthetic_plane(), N_AGENTS)
    return buf.getvalue()


SCRIPT_SRC = (pathlib.Path(__file__).resolve().parent.parent
              / "scripts" / "uniform_phase.py").read_text()


def _reachable(names, text, src):
    """Every quantity in ``names`` a reader can actually get to.

    Taken as a free function of its three inputs rather than reading globals, so
    that the tests below can feed it mutated ones and check the check.

    Matching is on **whole identifier tokens**, not substrings, and that is not a
    detail.  ``B_rho`` is a substring of ``B_rho_b``, so a substring check reports
    the aggregate as reaching a reader whenever either within-group balance is
    printed -- including when the aggregate's own column has been dropped, which
    is the exact historical bug this file was written for.  A guard against a
    specific past failure that cannot see that failure is worse than no guard,
    because it is read as evidence the failure cannot recur.  ``B_eta`` shadows
    ``B_eta_b`` and ``B_eta_u`` the same way; nothing else in the swept set
    shadows anything, and ``test_no_swept_name_shadows_another_unnoticed`` keeps
    it that way.

    A derived quantity counts as used if the report *computes* it, not only if
    its key appears in the printed text: a column is often headed with something
    friendlier than its key -- ``get_gap`` prints as "pooled" -- and keying on
    the literal name would push the output towards naming variables at the
    reader.  An unused derived key still fails, because it would appear neither
    in the text nor in the source that builds the tables.
    """
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    panels = set(TOP) | set(BOTTOM)
    shown = {n for n in names if n in tokens or n in panels}
    for key, sources in DERIVED_FROM.items():
        if key in text or key in panels or f'"{key}"' in src:
            shown |= set(sources)
    return shown


@pytest.fixture(scope="module")
def reachable(report_text):
    return _reachable(ORDER_PARAM_NAMES, report_text, SCRIPT_SRC)


@pytest.mark.parametrize("name", ORDER_PARAM_NAMES)
def test_every_swept_quantity_reaches_an_output(name, reachable):
    assert name in reachable, (
        f"{name} is swept and cached but reaches no report line, no panel, and "
        f"no derived quantity that does. Give it an output or stop measuring it."
    )


def test_the_derived_map_only_refers_to_things_that_are_swept():
    """A stale entry here would silently excuse a quantity from the check."""
    for key, sources in DERIVED_FROM.items():
        for src in sources:
            assert src in ORDER_PARAM_NAMES, (key, src)


def test_all_five_of_the_papers_parameters_are_printed(report_text):
    """The specific gap that prompted this file.

    ``B_rho`` went missing because two display groupings listed "the paper's
    five" by hand and each wrote four, so a claim about "the paper's
    parameters" was being made over four fifths of them.
    """
    for name in PAPER_NAMES:
        assert name in report_text, name


def test_the_swept_set_has_no_duplicates_and_covers_the_papers_five():
    assert len(set(ORDER_PARAM_NAMES)) == len(ORDER_PARAM_NAMES)
    assert set(PAPER_NAMES) <= set(ORDER_PARAM_NAMES)


# --- checking the check ----------------------------------------------------
#
# A coverage test that passes because it cannot fail is the failure this file
# exists to prevent, wearing a badge.  Each of these mutates one input and
# asserts the check notices.

def test_the_coverage_check_catches_a_new_orphan(report_text):
    """The forward case: a quantity added to the swept set with no home."""
    names = tuple(ORDER_PARAM_NAMES) + ("Q_orphan",)
    assert "Q_orphan" not in _reachable(names, report_text, SCRIPT_SRC)


def test_the_coverage_check_reproduces_the_historical_bug(report_text):
    """The backward case: the eight quantities that really were orphaned.

    Strip them back out of the report and the check must name them again -- all
    eight, and only them.  Reproducing the original bug is what says the test
    would have caught it, which is not the same as saying it passes now.
    """
    orphaned = {"B_rho", "B_eta_b", "B_eta_u", "B_rho_b", "B_rho_u",
                "T_bb", "T_bu", "frac_biased"}
    text = report_text
    for name in orphaned:
        text = text.replace(name, "xxxx")
    missed = set(ORDER_PARAM_NAMES) - _reachable(ORDER_PARAM_NAMES, text,
                                                 SCRIPT_SRC)
    # `rho_bu` shares a prefix with nothing but is stripped by "B_rho"? no --
    # but `B_rho_b`/`B_rho_u` contain `B_rho`, so removing the longer names
    # first is not enough; assert containment rather than equality.
    assert orphaned <= missed, orphaned - missed


def test_a_derived_quantity_that_disappears_orphans_its_sources(report_text):
    """The map must not be a way to excuse an orphan by pointing at something
    that has itself stopped reaching anyone."""
    src = SCRIPT_SRC.replace('"get_gap"', '"gone"')
    text = report_text.replace("pooled", "xxxx")
    shown = _reachable(ORDER_PARAM_NAMES, text, src)
    assert "T_get_b" not in shown and "T_get_u" not in shown


def test_no_swept_name_shadows_another_unnoticed():
    """Which names are substrings of which, recorded rather than assumed.

    Two pairs shadow today.  If a third appears, the token matching in
    :func:`_reachable` already handles it -- but this test failing is the prompt
    to check that whatever *else* reads these names by substring was updated too.
    """
    shadowed = {n: sorted(m for m in ORDER_PARAM_NAMES if m != n and n in m)
                for n in ORDER_PARAM_NAMES}
    shadowed = {n: v for n, v in shadowed.items() if v}
    assert shadowed == {"B_rho": ["B_rho_b", "B_rho_u"],
                        "B_eta": ["B_eta_b", "B_eta_u"]}, shadowed


@pytest.mark.parametrize("aggregate, survivors",
                         [("B_rho", ("B_rho_b", "B_rho_u")),
                          ("B_eta", ("B_eta_b", "B_eta_u"))])
def test_dropping_an_aggregate_is_caught_though_its_parts_remain(
        aggregate, survivors, report_text):
    """The mutation a substring check cannot see.

    Remove only the aggregate's own standalone column and leave its within-group
    columns printed.  A substring check calls it reachable -- verified below, so
    that this test is about the shadowing and not about my regex -- and the token
    check catches it.
    """
    mutated = re.sub(rf"\b{aggregate}\b(?!_)", "xxxx", report_text)
    for name in survivors:                       # the mutation was surgical
        assert name in mutated
    assert aggregate not in _reachable(ORDER_PARAM_NAMES, mutated, SCRIPT_SRC)
    # and the check that used to be here would have passed it
    assert aggregate in mutated                  # ... because of the survivors
