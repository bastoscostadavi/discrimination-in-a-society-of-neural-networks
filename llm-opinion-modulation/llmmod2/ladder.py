"""The nulling ladder: where a run of one-word verdicts turns over.

The instrument.  The model's verdict on a briefing is close to deterministic --
sampling the same briefing sixteen times usually gives sixteen identical answers,
and a psychometric curve swept over evidence is very nearly a step.  That rules
out reading a *frequency* as the measurement, which is what a stated probability
amounts to and what saturates.  It is, however, exactly what a nulling method
wants: a sharp step has a well-located position, and the position is the reading.

So nothing here measures how strongly the model believes something.  It measures
how much counter-evidence has to be laid against a belief before the verdict
turns over.  That quantity is in units of pieces of evidence, it is signed, and
it has no ceiling -- push the belief harder and the null point simply moves
further out.  The ``|h| <= 2.05`` clip that bounds the sibling experiment has no
analogue here.

The ladder is walked adaptively: one rung at the origin, geometric steps outward
until the verdict flips, then bisection into the bracket.  The rungs that result
are concentrated where the turnover is, which is where they carry information.
A probit fit over every rung visited then places the null point *between*
integers -- the individual verdicts are binary, but the crossing they bracket is
continuous, and it is the crossing that is reported.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from .llm import MAX_N, ask

#: How far the ladder may be walked before the measurement is called censored.
#: Chosen against ``OWN_TOTAL``: a belief that six pieces of evidence built
#: should not need more than twice that to overturn, and one that does is a
#: broken cell rather than a large number.
T_MAX = 12

#: Draws per rung.  Two requests at the API's cap of eight.  Sixteen is enough to
#: separate a rung that is turning over from one that is not; it is not trying to
#: resolve a probability, only to order the rungs.
DRAWS = 16

#: Bounds on the fitted slope.  A rung set that is entirely saturated would
#: otherwise send the slope to infinity; bounding it leaves the *position* of the
#: crossing, which is the quantity being measured, unaffected.
BETA_BOUNDS = (0.05, 20.0)


@dataclass
class Rung:
    t: int
    hits: int
    draws: int

    @property
    def p(self):
        return self.hits / self.draws if self.draws else float("nan")


@dataclass
class NullPoint:
    t_star: float
    beta: float
    rungs: list = field(default_factory=list)
    censored: bool = False

    @property
    def leaning(self):
        """Belief before the consignment, in pieces of evidence.

        The null point is how much evidence must be added to reach indifference,
        so the belief already held is its negative.
        """
        return -self.t_star

    def as_dict(self):
        return {"t_star": self.t_star, "beta": self.beta,
                "censored": self.censored,
                "rungs": [[r.t, r.hits, r.draws] for r in self.rungs]}


def _probit_fit(rungs):
    """Maximum likelihood ``(t_star, beta)`` for ``p(t) = Phi(beta (t - t*))``.

    Binomial likelihood rather than a fit to transformed proportions: a rung at
    zero or one is ordinary data here, whereas a probit of the proportion would
    be infinite and would have to be clipped -- which is the thing this
    experiment exists to avoid.
    """
    t = np.array([r.t for r in rungs], float)
    hit = np.array([r.hits for r in rungs], float)
    n = np.array([r.draws for r in rungs], float)

    def nll(theta):
        t_star, log_beta = theta
        p = ndtr(np.exp(log_beta) * (t - t_star))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -float(hit @ np.log(p) + (n - hit) @ np.log1p(-p))

    lo, hi = float(t.min()) - 2.0, float(t.max()) + 2.0
    best = None
    for t0 in np.linspace(lo, hi, 9):
        for b0 in (-0.7, 0.0, 1.0):
            r = minimize(nll, [t0, b0], method="L-BFGS-B",
                         bounds=[(lo, hi), (np.log(BETA_BOUNDS[0]),
                                            np.log(BETA_BOUNDS[1]))])
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(np.exp(best.x[1]))


def _hits(answers, target):
    return sum(1 for a in answers if a.get("answer") == target)


#: Offsets tried when walking outward from the origin, in order.  Geometric so
#: that a strongly held belief is bracketed in four rungs rather than twelve, and
#: ending exactly on ``T_MAX`` so the ladder is walked to its stated end.
SCHEDULE = (1, 2, 4, 8, T_MAX)


def measure_null(render, schema, target, system, *, draws=DRAWS, t_max=T_MAX,
                 max_rungs=8, **ask_kw):
    """Find the ladder value at which the verdict turns over.

    ``render(t)`` builds the briefing for ladder value ``t``; ``target`` is the
    entity name whose frequency is being tracked, and ``p(t)`` is increasing in
    ``t`` by construction because ``t`` is signed towards it.

    Returns a :class:`NullPoint`.  ``censored`` marks a cell whose verdict never
    turned over inside the ladder; such a cell is kept in the data and excluded
    from the fits rather than quietly clipped to the edge.
    """
    seen = {}

    def evaluate(t):
        if t in seen:
            return seen[t]
        user = render(t)
        sizes, left = [], draws
        while left > 0:
            sizes.append(min(MAX_N, left))
            left -= sizes[-1]
        # The batches of one rung are independent, and the rungs of one ladder
        # are not -- each is chosen from the last.  So the only concurrency
        # available inside a measurement is here, and taking it halves the
        # latency of every rung.
        with ThreadPoolExecutor(len(sizes)) as pool:
            batches = pool.map(
                lambda i: ask(system, user, schema, nonce=i, n=sizes[i], **ask_kw),
                range(len(sizes)))
            answers = [a for batch in batches for a in batch]
        rung = Rung(t=t, hits=_hits(answers, target), draws=len(answers))
        seen[t] = rung
        return rung

    origin = evaluate(0)
    above = origin.p > 0.5
    direction = -1 if above else 1
    prev, bracket = 0, None
    for step in SCHEDULE:
        if step > t_max or len(seen) >= max_rungs:
            break
        t = direction * step
        if (evaluate(t).p > 0.5) != above:
            bracket = tuple(sorted((prev, t)))
            break
        prev = t

    censored = bracket is None
    if not censored:
        lo, hi = bracket
        while hi - lo > 1 and len(seen) < max_rungs:
            mid = (lo + hi) // 2
            if (evaluate(mid).p > 0.5) == (evaluate(lo).p > 0.5):
                lo = mid
            else:
                hi = mid

    rungs = [seen[t] for t in sorted(seen)]
    t_star, beta = _probit_fit(rungs)
    return NullPoint(t_star=t_star, beta=beta, rungs=rungs, censored=censored)
