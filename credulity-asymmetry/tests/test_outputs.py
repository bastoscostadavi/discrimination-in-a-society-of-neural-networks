"""Every swept quantity must reach a reader, and this check must be able to fail.

A quantity can be measured, cached in every ``.npz``, given a label, a colour map
and a range, described in a docstring, and displayed nowhere.  That happened here
to ``B_rho_A`` and ``B_rho_B``: two of the twelve, carrying the one statement that
the two classes are reorganized in the trust sector and left alike in the opinion
sector, and no output in the package would have told anyone.  The cause is display
groupings written out by hand -- ``paper = ("R_muc", "R_cw", "R_wmu", "B_eta")``
is a four-tuple that looks complete and silently drops the fifth, under prose
claiming all five.

Two things about *this* file matter as much as the coverage it asserts.

**It checks what the report prints, not what the source says.**  A test keyed to
source text passes only while the literals are literals, and breaks the moment
they are replaced by the constant that fixes the bug, which is a mild pressure
against fixing it.

**It matches column headings exactly, and is mutation-tested.**  The first version
used substring containment, and could not detect the very bug it was written for:
``"B_rho" in text`` is true whenever ``B_rho^AA`` is printed, so dropping the
aggregate ``B_rho`` column left the check green.  A guard against a specific
historical failure that cannot see that failure is worse than no guard, because it
is read as evidence.  So the reachability logic is a free function of its inputs
rather than something reading globals -- which is what makes it possible to feed
counterfactuals -- and the tests at the bottom mutate the report and require the
check to notice.
"""

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

from credfield.order_params import ORDER_PARAM_NAMES, PAPER_NAMES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

#: Plain-text column heading for each quantity whose printed name differs from its
#: key.  Exact headings, because matching is exact.
HEADINGS = {"B_eta_A": "B_eta^AA", "B_eta_B": "B_eta^BB",
            "B_rho_A": "B_rho^AA", "B_rho_B": "B_rho^BB"}

#: Quantities that reach a reader only through something derived from them.  Empty
#: here, deliberately: every swept quantity is printed under its own heading, so
#: there is no derivation to excuse an orphan with.  Kept as an explicit empty map
#: rather than omitted, so that adding an entry is a visible decision and
#: ``test_no_quantity_relies_on_a_derived_alias`` fails when one appears without
#: the corresponding reachability rule.
DERIVED_FROM = {}


def heading(name):
    """The exact column heading a quantity is printed under."""
    return HEADINGS.get(name, name)


def header_tokens(text):
    """The table's column headings, as exact tokens.

    Splitting the header line on whitespace and column separators is what makes
    the match exact.  Substring search over the whole report cannot distinguish
    ``B_rho`` from ``B_rho^AA``, and those are different columns carrying
    different numbers.
    """
    lines = [l for l in text.splitlines() if l.strip().startswith("condition")]
    if not lines:
        return set()
    return {t for t in re.split(r"[\s|]+", lines[0]) if t}


def unreachable(names, text, derived=None):
    """Which of ``names`` no reader of ``text`` could find.  Pure, so testable."""
    derived = DERIVED_FROM if derived is None else derived
    tokens = header_tokens(text)
    missing = []
    for name in names:
        if heading(name) in tokens:
            continue
        alias = derived.get(name)
        if alias and alias in tokens:
            continue
        missing.append(name)
    return missing


def _table_output():
    """Run the invisibility table on a tiny batch and capture what it prints."""
    import invisibility
    from credfield.config import get_preset

    preset = get_preset("quick")
    model = preset.model.with_(n_dim=5, n_issues=2, interactions_per_channel=2.0)
    preset = preset.__class__(**{**preset.__dict__, "model": model,
                                "demo_agents": 8, "demo_runs": 2})
    batch, reps = invisibility.run(preset, verbose=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        invisibility.table(batch, reps)
    return buf.getvalue()


@pytest.fixture(scope="module")
def table_text():
    return _table_output()


# --- the coverage assertion ---------------------------------------------

def test_every_swept_quantity_is_printed_by_the_table(table_text):
    missing = unreachable(ORDER_PARAM_NAMES, table_text)
    assert not missing, (
        f"swept but never shown to anyone: {missing}. Either display it or stop "
        f"sweeping it; a quantity in the cache and in no output is a measurement "
        f"nobody can read.")


def test_all_five_published_parameters_are_shown_not_four(table_text):
    """The specific regression: a hand-written four-tuple for a set of five."""
    assert len(PAPER_NAMES) == 5
    assert not unreachable(PAPER_NAMES, table_text)


def test_the_table_covers_every_field_component(table_text):
    from credfield.fields import COMPONENTS
    for comp in COMPONENTS:
        assert f"{comp} = 1" in table_text, comp


def test_every_swept_quantity_has_a_label_a_range_and_a_map():
    from credfield.plotting import CMAPS, LABELS, RANGES
    for name in ORDER_PARAM_NAMES:
        assert name in LABELS and name in RANGES and name in CMAPS, name


def test_no_quantity_relies_on_a_derived_alias():
    """Nothing here is reachable only via a derivation, so the map stays empty.

    If an entry appears, the alias must itself be a printed heading -- otherwise
    the map is a way to excuse an orphan by pointing at something that has itself
    stopped reaching anyone.
    """
    assert DERIVED_FROM == {}


# --- mutation tests: can the check above fail? --------------------------

def test_the_check_catches_a_quantity_with_no_home(table_text):
    assert unreachable(list(ORDER_PARAM_NAMES) + ["Q_orphan"], table_text) \
        == ["Q_orphan"]


@pytest.mark.parametrize("name", ["B_rho", "B_eta"])
def test_the_check_catches_dropping_an_aggregate_whose_name_is_a_prefix(
        table_text, name):
    """The substring failure, as a test rather than as a memory.

    ``B_rho`` is a prefix of the printed ``B_rho^AA``, so a containment check
    stays green when the aggregate column is dropped -- which is exactly the
    historical bug.  Removing just that column from the header must be noticed,
    and the within-class columns must survive the removal so the test is about
    the aggregate and not about the string munging.
    """
    mutated = re.sub(rf"(?<= ){re.escape(name)}(?= )", "x" * len(name),
                     table_text, count=1)
    tokens = header_tokens(mutated)
    assert name not in tokens                       # the column really is gone
    assert f"{name}^AA" in tokens                   # and its namesakes remain
    missing = unreachable(ORDER_PARAM_NAMES, mutated)
    assert name in missing


def test_the_check_catches_dropping_a_within_class_column(table_text):
    """And the original omission, reproduced.

    Containment rather than equality on the names: stripping ``B_rho^AA`` by
    string replacement can disturb neighbouring headings, so requiring the set to
    be exactly the two would be testing the mutation rather than the check.
    """
    mutated = table_text.replace("B_rho^AA", "xxxxxxxx").replace(
        "B_rho^BB", "yyyyyyyy")
    missing = unreachable(ORDER_PARAM_NAMES, mutated)
    assert {"B_rho_A", "B_rho_B"} <= set(missing)


def test_the_check_is_clean_on_the_real_report(table_text):
    """The other half of a mutation test: no false positives unmutated."""
    assert unreachable(ORDER_PARAM_NAMES, table_text) == []
    assert len(header_tokens(table_text)) > len(ORDER_PARAM_NAMES)


def test_a_derived_alias_cannot_excuse_an_orphan_it_no_longer_reaches():
    """If the alias is not itself printed, the quantity is still unreachable."""
    text = "condition   R_wmu   R_muc\n"
    assert unreachable(["B_eta_A"], text, derived={"B_eta_A": "atomization"}) \
        == ["B_eta_A"]
    # and when the alias *is* printed, it excuses it
    text2 = "condition   R_wmu   atomization\n"
    assert unreachable(["B_eta_A"], text2,
                       derived={"B_eta_A": "atomization"}) == []


def test_atomization_is_still_what_the_composite_draws():
    """The derived quantity the figures use, checked against its definition."""
    import cred_asymmetry as ca
    n_f, n_s = 4, 5
    data = {"B_eta_A": np.full((n_f, n_s), 0.8),
            "B_eta_B": np.full((n_f, n_s), -0.4)}
    np.testing.assert_allclose(ca.atomization(data), 0.6)
