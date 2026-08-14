"""A batch of societies of interacting neural agents, evolved in lockstep.

Each agent carries two sectors:

* an **ideological** sector --- a weight vector ``w`` in ``R^K`` with covariance
  ``C``, giving its opinion ``sign(w . x)`` on an issue ``x``;
* an **affective** sector --- for every *other* agent, a distrust ``mu`` with
  variance ``V``.

At each step an issue, an emitter and a receiver are drawn.  The emitter states
``sigma_e = sign(w_e . x)``; the receiver forms two scaled fields and updates all
four of its quantities through the modulation functions:

    gamma_C = sqrt(1 + x.C_r x)             gamma_V = sqrt(1 + V)
    h_w     = (w_r . x) sigma_e / gamma_C + D    h_mu = mu / gamma_V

    w_r += (F_w / gamma_C) sigma_e C_r x
    C_r += (F_C / gamma_C^2) (C_r x)(C_r x)^T
    mu  += (F_mu / gamma_V) V
    V   += (F_V / gamma_V^2) V^2

``D`` is the discrimination field, zero for non-discriminating receivers.

The dynamics anneals
--------------------
``F_C`` and ``F_V`` are negative almost everywhere, so both uncertainties shrink
monotonically: the society does not approach a stationary state, it slows down.
**When you measure is therefore a parameter of the experiment, not an incidental
detail**, and every quantity this package reports is conditional on the
interaction count.  This is easy to forget and it is the assumption most likely
to be challenged, so it is stated here, in the paper's setup section, and tested
by a sensitivity sweep.

Layout, and why it matters
--------------------------
``SocietyBatch`` evolves ``R`` independent societies at once --- typically many
grid points of a phase diagram at one replicate index.  State is stored
**agent-major**::

    w: (N, R, K)      C: (N, R, K, K)      mu, V: (N, N, R)

and the interaction schedule is *shared* across the batch, so ``C[r]`` and
``mu[r, e]`` are contiguous views rather than gathers and one numpy call
advances all ``R`` societies.  The alternatives are 3-5x slower: a per-society
Python loop, or a run-major layout whose fancy indexing copies the whole
covariance tensor every step.

Sharing the schedule couples the *order* of interactions across the batch, never
the content --- weights, distrust, agendas, class labels and the discriminator
mask are drawn per society.  See ``seeds.derived_seed`` for why this is a
variance reduction across pixels rather than a bias in the seed-to-seed spread.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .discrimination import FieldSpec, build_field
from .modulation import Z_FLOOR, modulation
from .seeds import stream

__all__ = ["SocietyBatch", "V_FLOOR", "RULES", "FREEZABLE"]

#: Floor on the affective variance.
#:
#: ``F_V < 0`` shrinks ``V`` monotonically towards zero, so on a channel that
#: interacts often enough ``V`` can underflow to a non-positive value.  The model
#: does not prescribe this floor.
V_FLOOR = 1e-12

RULES = ("entropic", "hebbian", "perceptron")
FREEZABLE = ("trust", "opinion")


class SocietyBatch:
    """``R`` independent societies of ``N`` agents, advanced together.

    Construct with :meth:`from_keys`, which derives every random stream from the
    run keys.  The constructor here takes already-built state and is mostly for
    tests.
    """

    def __init__(
        self,
        n_agents,
        n_dim,
        n_issues,
        keys,
        field_specs,
        master,
        rule="entropic",
        freeze=(),
        class_balance=0.5,
        dtype=np.float64,
        z_floor=Z_FLOOR,
        v_floor=V_FLOOR,
        step_size=None,
        margin=0.0,
    ):
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        for f in freeze:
            if f not in FREEZABLE:
                raise ValueError(f"freeze entries must be in {FREEZABLE}, got {f!r}")
        keys = list(keys)
        specs = list(field_specs)
        if len(specs) != len(keys):
            raise ValueError("need one FieldSpec per run key")
        # The shared schedule is only legitimate if every society in the batch
        # draws it from the same stream.
        if len({(k.crn_group, k.init) for k in keys}) > 1:
            raise ValueError(
                "a batch must hold one (crn_group, init) pair; "
                "mixing replicates within a batch would share schedules across "
                "replicates and understate the seed-to-seed spread"
            )

        self.N, self.K, self.P, self.R = int(n_agents), int(n_dim), int(n_issues), len(keys)
        self.keys, self.field_specs, self.master = keys, specs, master
        self.rule, self.freeze = rule, tuple(freeze)
        self.dtype, self.z_floor, self.v_floor = dtype, z_floor, v_floor
        self.margin = float(margin)
        self.n_steps_taken = 0
        self.n_psd_clips = 0
        self.n_zfloor_hits = 0

        N, K, P, R = self.N, self.K, self.P, self.R

        self.w = np.empty((N, R, K), dtype=dtype)
        self.C = np.empty((N, R, K, K), dtype=dtype)
        self.mu = np.empty((N, N, R), dtype=dtype)
        self.V = np.empty((N, N, R), dtype=dtype)
        self.X = np.empty((P, R, K), dtype=dtype)
        self.D = np.zeros((N, N, R), dtype=dtype)
        self.class_of = np.empty((N, R), dtype=np.int8)
        self.kappa = np.empty((N, R), dtype=dtype)
        self.discriminates = np.zeros((N, R), dtype=bool)

        n_a = int(round(class_balance * N))
        eye = np.eye(K, dtype=dtype)

        for r, (key, spec) in enumerate(zip(keys, specs)):
            rng_cls = stream(key, "classes", master)
            rng_agenda = stream(key, "agenda", master)
            rng_mask = stream(key, "mask", master)
            rng_field = stream(key, "field", master)
            rng_init = stream(key, "init", master)

            # Classes: a fixed split, then permuted so class membership is not
            # confounded with agent index.
            cls = np.zeros(N, dtype=np.int8)
            cls[n_a:] = 1
            cls = cls[rng_cls.permutation(N)]
            self.class_of[:, r] = cls
            self.kappa[:, r] = np.where(cls == 0, 1.0, -1.0)

            # Uninformative Gaussian prior; initial distrust uniform on [-1, 1],
            # which starts the society with as many frustrated triples as
            # unfrustrated ones.
            self.w[:, r, :] = rng_init.normal(size=(N, K))
            self.C[:, r] = eye
            self.mu[:, :, r] = rng_init.uniform(-1.0, 1.0, size=(N, N))
            self.V[:, :, r] = 1.0

            Xr = rng_agenda.normal(size=(P, K))
            Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
            self.X[:, r, :] = Xr

            D, disc = build_field(spec, cls, rng_mask, rng_field, dtype=dtype)
            self.D[:, :, r] = D
            self.discriminates[:, r] = disc

        self._rng_schedule = stream(keys[0], "schedule", master)

        # Step sizes for the non-entropic rules.  These must be calibrated
        # against the entropic rule's mean |dw| before the comparison means
        # anything -- see scripts/calibrate_rules.py.
        self.step_size = step_size
        if rule != "entropic" and step_size is None:
            raise ValueError(
                f"rule={rule!r} needs an explicit step_size; an uncalibrated "
                "step size makes the ablation a statement about learning rate "
                "rather than about the rule"
            )

        self._update_opinion = getattr(self, f"_opinion_{rule}")
        if "opinion" in self.freeze:
            self._update_opinion = self._opinion_frozen
        self._update_trust = (
            self._trust_frozen if "trust" in self.freeze else self._trust_entropic
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_keys(cls, model, keys, field_specs, master, **kw):
        """Build from a :class:`~socsim.config.ModelConfig`."""
        return cls(
            n_agents=model.n_agents,
            n_dim=model.n_dim,
            n_issues=model.n_issues,
            keys=keys,
            field_specs=field_specs,
            master=master,
            rule=model.rule,
            freeze=model.freeze,
            class_balance=model.class_balance,
            dtype=model.numpy_dtype(),
            step_size=model.step_size,
            margin=model.margin,
            **kw,
        )

    # -- diagnostics ---------------------------------------------------
    @property
    def n_interactions_per_channel(self):
        return self.n_steps_taken / (self.N * (self.N - 1))

    def memory_bytes(self):
        return sum(
            a.nbytes for a in (self.w, self.C, self.mu, self.V, self.X, self.D)
        )

    # -- dynamics ------------------------------------------------------
    def run(self, n_steps, measure_at=None, measure_fn=None):
        """Advance ``n_steps`` interactions, measuring at the given step counts."""
        out = []
        targets = list(measure_at or ())
        next_target = targets.pop(0) if targets else None
        N, P = self.N, self.P
        rng = self._rng_schedule

        # Draw the schedule up front: cheap, and keeps per-step generator calls
        # out of the inner loop.  Chunked so memory stays bounded for long runs.
        remaining = int(n_steps)
        while remaining > 0:
            block = min(remaining, 1_000_000)
            recv = rng.integers(N, size=block)
            emit = rng.integers(N - 1, size=block)
            emit += emit >= recv
            issue = rng.integers(P, size=block)
            for t in range(block):
                self._interact(int(recv[t]), int(emit[t]), self.X[issue[t]])
                self.n_steps_taken += 1
                if next_target is not None and self.n_steps_taken >= next_target:
                    out.append(measure_fn(self))
                    next_target = targets.pop(0) if targets else None
            remaining -= block
        return out

    def _interact(self, r, e, x):
        """One interaction. All accesses are contiguous views."""
        w_r, C_r = self.w[r], self.C[r]  # (R, K), (R, K, K)

        sigma = np.sign(np.einsum("rk,rk->r", self.w[e], x))
        sigma[sigma == 0] = 1.0

        Cx = np.einsum("rij,rj->ri", C_r, x)  # (R, K)
        xCx = np.einsum("rk,rk->r", x, Cx)
        gamma_C = np.sqrt(1.0 + xCx)

        V = self.V[r, e]
        gamma_V = np.sqrt(1.0 + V)

        h_w = np.einsum("rk,rk->r", w_r, x) * sigma / gamma_C + self.D[r, e]
        h_mu = self.mu[r, e] / gamma_V

        F_w, F_C, F_mu, F_V, n_floor = modulation(
            h_w, h_mu, self.z_floor, count_floor=True
        )
        self.n_zfloor_hits += n_floor

        self._update_opinion(r, x, sigma, Cx, xCx, gamma_C, F_w, F_C, h_w, h_mu)
        self._update_trust(r, e, V, gamma_V, F_mu, F_V)

    # -- opinion sector ------------------------------------------------
    #
    # Every rule receives the *same* ``h_w``, which already carries the
    # discrimination field.  That is the point of the ablation: the intervention
    # is literally the same object in all three, so a difference in outcome is a
    # difference in what the rule does with it.  ``tests/test_rules.py`` asserts
    # it rather than trusting the convention.
    def _opinion_entropic(self, r, x, sigma, Cx, xCx, gamma_C, F_w, F_C, h_w, h_mu):
        w_r, C_r = self.w[r], self.C[r]
        w_r += ((F_w * sigma / gamma_C)[:, None]) * Cx
        a = self._project_psd(F_C / (gamma_C * gamma_C), xCx)
        C_r += a[:, None, None] * (Cx[:, :, None] * Cx[:, None, :])

    def _opinion_hebbian(self, r, x, sigma, Cx, xCx, gamma_C, F_w, F_C, h_w, h_mu):
        """Trust-signed Hebb: no surprise gate, no annealing.

        Note ``F_w = tau * g(h_w) / Z`` with ``tau = 1 - 2 Phi(h_mu)`` the trust
        score, so the entropic opinion update *is* trust-signed Hebb times a
        surprise gate.  Dropping the gate ablates exactly one factor.

        A plain Hebbian update ``w += eta sigma_e x`` would not depend on ``h_w``
        at all, so the discrimination field could provably do nothing and a flat
        result would carry no information.  Keeping the trust sign is what makes
        this control informative: the field still reaches the opinion sector,
        through ``tau``, just without the evidence weighting.
        """
        tau = 1.0 - 2.0 * ndtr(h_mu)
        self.w[r] += (self.step_size * tau * sigma)[:, None] * x
        self._project_norm(r)

    def _opinion_perceptron(self, r, x, sigma, Cx, xCx, gamma_C, F_w, F_C, h_w, h_mu):
        """Margin-gated, trust-signed perceptron.

        A distrusted source's stated opinion is inverted before use, the discrete
        counterpart of the entropic rule's ability to learn the opposite of what
        a distrusted agent says.  The margin test uses the *perceived* field
        ``h_w``, so the discrimination field enters here exactly as it does in the
        entropic rule.
        """
        tau = 1.0 - 2.0 * ndtr(h_mu)
        s = np.sign(tau)
        s[s == 0] = 1.0
        # h_w is the receiver's stability against the emitter's claim; flipping
        # by the trust sign gives the stability against the pseudo-label.
        active = s * h_w < self.margin
        if np.any(active):
            y = s * sigma
            step = np.where(active, self.step_size * np.abs(tau) * y, 0.0)
            self.w[r] += step[:, None] * x
            self._project_norm(r)

    def _opinion_frozen(self, r, x, sigma, Cx, xCx, gamma_C, F_w, F_C, h_w, h_mu):
        """Opinions held fixed; only the trust sector adapts."""
        return

    # -- affective sector ----------------------------------------------
    def _trust_entropic(self, r, e, V, gamma_V, F_mu, F_V):
        self.mu[r, e] += (F_mu / gamma_V) * V
        np.maximum(
            V + (F_V / (gamma_V * gamma_V)) * V * V, self.v_floor, out=self.V[r, e]
        )

    def _trust_frozen(self, r, e, V, gamma_V, F_mu, F_V):
        """Trust held fixed.

        Note the frozen *value* matters and the obvious choice is degenerate:
        at ``mu = 0`` we get ``h_mu = 0``, hence ``1 - 2 Phi(0) = 0``, hence
        ``F_w = 0`` identically --- the opinion sector would not learn at all and
        the control would be vacuous.  The runner therefore freezes at the
        random initialisation, and ``tests/test_controls.py`` pins the
        degeneracy so it cannot be reintroduced.
        """
        return

    # -- helpers -------------------------------------------------------
    def _project_norm(self, r):
        """Keep ``||w|| = sqrt(K)`` for the rules that have no annealing.

        Without the covariance update there is nothing to arrest the growth of
        ``||w||``, so ``h_w`` would diverge, the trust sector would saturate, and
        the alternative rules would lose to the entropic one for reasons having
        nothing to do with class.  Every observable used in the paper is a
        cosine, hence invariant to this rescaling.
        """
        w_r = self.w[r]
        norm = np.linalg.norm(w_r, axis=1, keepdims=True)
        np.divide(w_r, np.maximum(norm, 1e-300), out=w_r)
        w_r *= np.sqrt(self.K)

    def _project_psd(self, a, xCx, margin=1e-9):
        """Keep the covariance update from pushing ``C`` out of PSD.

        ``C -> C + a (Cx)(Cx)^T`` stays positive semi-definite exactly when
        ``1 + a x.Cx >= 0``, the binding direction being ``x`` itself.  ``F_C`` is
        unbounded below near the dissonant corners, so ``a`` is clipped to that
        boundary.  The clip only ever *reduces* the magnitude of a step the exact
        algorithm would have taken, and the counter records how often it binds.
        """
        lo = -(1.0 - margin) / np.maximum(xCx, 1e-300)
        n_bad = int(np.count_nonzero(a < lo))
        if n_bad:
            self.n_psd_clips += n_bad
            a = np.maximum(a, lo)
        return a
