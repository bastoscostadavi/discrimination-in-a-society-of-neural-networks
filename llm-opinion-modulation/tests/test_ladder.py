"""The ladder has to place a crossing it is shown, and admit one it is not."""

import numpy as np
import pytest
from scipy.special import ndtr

from llmmod2.ladder import Rung, _probit_fit, measure_null
from llmmod2.prompts import SYSTEM

SCHEMA = {"type": "object",
          "properties": {"answer": {"type": "string", "enum": ["a", "b"]}},
          "required": ["answer"], "additionalProperties": False}


def _rungs(t_star, beta, ts, draws=16, seed=0):
    rng = np.random.default_rng(seed)
    return [Rung(t=t, hits=int(rng.binomial(draws, ndtr(beta * (t - t_star)))),
                 draws=draws) for t in ts]


@pytest.mark.parametrize("t_star", [-3.4, -1.0, 0.0, 0.7, 2.5])
def test_probit_fit_recovers_the_crossing(t_star):
    rungs = _rungs(t_star, 1.5, range(-6, 7))
    fitted, beta = _probit_fit(rungs)
    assert fitted == pytest.approx(t_star, abs=0.5)
    assert beta > 0.0


def test_a_saturated_ladder_still_places_the_crossing():
    """Every rung at zero or one -- the common case, and the one a probit of the
    proportion could not handle at all."""
    rungs = [Rung(t=t, hits=16 if t >= 2 else 0, draws=16) for t in range(-4, 5)]
    fitted, _ = _probit_fit(rungs)
    assert 1.0 <= fitted <= 2.0


class _Fake:
    """A deterministic model that answers from a threshold, for the walk."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.seen = []

    def __call__(self, system, user, schema, nonce=0, n=8, **kw):
        t = int(user)
        self.seen.append(t)
        return [{"answer": "a" if t > self.threshold else "b"}] * n


@pytest.mark.parametrize("threshold", [-5, -1, 0, 3, 6])
def test_walk_brackets_the_threshold(monkeypatch, threshold):
    fake = _Fake(threshold)
    monkeypatch.setattr("llmmod2.ladder.ask", fake)
    point = measure_null(str, SCHEMA, "a", SYSTEM)
    assert not point.censored
    assert point.t_star == pytest.approx(threshold + 0.5, abs=1.0)
    assert point.leaning == pytest.approx(-point.t_star)


def test_a_belief_outside_the_ladder_is_censored_not_clipped(monkeypatch):
    """Nothing is quietly pinned to the edge; the cell is marked and kept."""
    monkeypatch.setattr("llmmod2.ladder.ask", _Fake(10_000))
    point = measure_null(str, SCHEMA, "a", SYSTEM)
    assert point.censored
