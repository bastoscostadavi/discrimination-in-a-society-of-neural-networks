"""A batch of societies of entropic-dynamics neural-network agents.

Each agent is a single-layer perceptron carrying

  * an opinion sector: weight vector ``w`` (K,) and covariance ``C`` (K, K),
  * an trust sector: for every other agent, a distrust ``mu`` and its
    variance ``V``.

At each time step an issue, an emitter and a receiver are drawn.  The emitter
states its opinion ``sigma_e = sign(w_e . x)``; the receiver updates all four of
its quantities through the modulation functions of :mod:`ednna.modulation`:

    gamma_C = sqrt(1 + x.C_r x)             gamma_V = sqrt(1 + V)
    h_w     = (w_r . x) sigma_e / gamma_C + D    h_mu = mu / gamma_V

    w_r += (F_w / gamma_C) sigma_e C_r x
    C_r += (F_C / gamma_C^2) (C_r x)(C_r x)^T
    mu  += (F_mu / gamma_V) V
    V   += (F_V / gamma_V^2) V^2

where ``D`` is the prejudice field (zero for non-prejudiced receivers).

Layout and why it matters
-------------------------
``SocietyBatch`` evolves ``R`` independent societies at once — typically the R
pixels of a phase diagram, each with its own ``(d, f_d)``.  State is stored
**agent-major**::

    w: (N, R, K)      C: (N, R, K, K)      mu, V: (N, N, R)

and the interaction schedule (which receiver, which emitter, which issue index)
is *shared* across the batch.  Together these mean every access in the inner
loop, ``C[r]`` and ``mu[r, e]``, is a contiguous view rather than a gather, and
one numpy call advances all R societies.  Measured throughput is ~0.6M
agent-updates per second per core; the obvious alternatives (per-run Python
loop, or a run-major layout with fancy indexing) are 3-5x slower, the latter
because indexing copies the whole covariance tensor every step.

Sharing the schedule couples the *order* of interactions across runs but not
their content: initial weights, initial distrust, agendas and the assignment of
which agents discriminate are all drawn independently per run.  Neighbouring
pixels of a phase diagram therefore act as common-random-number pairs, which if
anything reduces the visual noise between them.  Pass
``shared_schedule=False`` for fully independent schedules at ~3x the cost.
"""

from __future__ import annotations

import numpy as np

from .discrimination import field_matrix
from .modulation import Z_FLOOR, modulation

__all__ = ["SocietyBatch", "V_FLOOR"]

#: Floor on the trust variance V.
#:
#: F_V < 0 shrinks V monotonically towards zero (the receiver becomes ever more
#: certain of its distrust), so V can underflow to a non-positive value after
#: enough interactions on the same channel.  The draft does not mention this.
V_FLOOR = 1e-12


class SocietyBatch:
    """``R`` independent societies of ``N`` agents, evolved in lockstep.

    Parameters
    ----------
    n_agents, n_dim, n_issues
        ``N``, ``K``, ``P``: society size, embedding dimension of an issue, and
        number of issues in the agenda.
    d, f_d
        Arrays of length ``R`` (or scalars, broadcast): the prejudice field
        and the fraction of prejudiced agents for each society in the batch.
    case
        Which discrimination case of the draft's Table I (1-6); default 6.
    seed
        Seed for this batch's initial conditions and schedule.
    literal_draft_sign
        Passed to :func:`ednna.discrimination.field_matrix`; see
        ``docs/prejudice-field-sign.md``.
    dtype
        ``np.float64`` (default) or ``np.float32``.  float32 halves memory
        traffic, which is the bottleneck, for a ~1.7x speedup.
    """

    def __init__(
        self,
        n_agents,
        n_dim,
        n_issues,
        d=0.0,
        f_d=0.0,
        case=6,
        seed=0,
        literal_draft_sign=False,
        dtype=np.float64,
        shared_schedule=True,
        z_floor=Z_FLOOR,
        v_floor=V_FLOOR,
    ):
        d = np.atleast_1d(np.asarray(d, dtype=dtype))
        f_d = np.atleast_1d(np.asarray(f_d, dtype=dtype))
        n_runs = max(d.size, f_d.size)
        if d.size == 1:
            d = np.repeat(d, n_runs)
        if f_d.size == 1:
            f_d = np.repeat(f_d, n_runs)
        if d.size != n_runs or f_d.size != n_runs:
            raise ValueError("d and f_d must have the same length (or be scalars)")

        self.N = int(n_agents)
        self.K = int(n_dim)
        self.P = int(n_issues)
        self.R = int(n_runs)
        self.d = d
        self.f_d = f_d
        self.case = case
        self.dtype = dtype
        self.shared_schedule = bool(shared_schedule)
        self.z_floor = z_floor
        self.v_floor = v_floor
        self.n_steps_taken = 0
        #: how many times the positive-definiteness projection on C was needed
        self.n_psd_clips = 0

        rng = np.random.default_rng(seed)
        self.rng = rng
        N, K, P, R = self.N, self.K, self.P, self.R

        # --- classes: the society is split in half, A then B ---------------
        self.class_of = np.zeros(N, dtype=np.int8)
        self.class_of[N // 2 :] = 1
        self.kappa = np.where(self.class_of == 0, 1.0, -1.0).astype(dtype)

        # --- initial conditions --------------------------------------------
        # Uninformative Gaussian prior per agent: w ~ N(0, I), C = I, V = 1.
        # Initial distrust is uniform on [-1, 1], which starts the society with
        # as many frustrated triples as unfrustrated ones.
        self.w = rng.normal(size=(N, R, K)).astype(dtype)
        self.C = np.zeros((N, R, K, K), dtype=dtype)
        self.C[:, :] = np.eye(K, dtype=dtype)
        self.mu = rng.uniform(-1.0, 1.0, size=(N, N, R)).astype(dtype)
        self.V = np.ones((N, N, R), dtype=dtype)

        # --- the agenda: P issues per run, embedded as unit vectors ---------
        X = rng.normal(size=(P, R, K)).astype(dtype)
        X /= np.linalg.norm(X, axis=2, keepdims=True)
        self.X = X

        # --- who discriminates, and by how much ----------------------------
        # discriminates[i, run] is drawn independently per agent and run with
        # probability f_d[run]; the field a prejudiced receiver applies
        # depends on its own class and the emitter's.
        self.discriminates = rng.random((N, R)) < f_d[None, :]
        self.D = np.zeros((N, N, R), dtype=dtype)
        for run in range(R):
            M = field_matrix(d[run], case=case, literal_draft=literal_draft_sign)
            block = M[np.ix_(self.class_of, self.class_of)]  # (N, N)
            self.D[:, :, run] = block * self.discriminates[:, run][:, None]

    # -- diagnostics ------------------------------------------------------
    @property
    def n_interactions_per_channel(self):
        """Mean number of interactions each ordered pair has had."""
        return self.n_steps_taken / (self.N * (self.N - 1))

    def memory_bytes(self):
        return self.w.nbytes + self.C.nbytes + self.mu.nbytes + self.V.nbytes + self.X.nbytes + self.D.nbytes

    # -- dynamics ---------------------------------------------------------
    def step(self):
        """Advance every society in the batch by one interaction."""
        N = self.N
        r = int(self.rng.integers(N))
        e = int(self.rng.integers(N - 1))
        if e >= r:
            e += 1  # uniform over emitters != receiver
        p = int(self.rng.integers(self.P))
        self._interact(r, e, self.X[p])
        self.n_steps_taken += 1

    def run(self, n_steps, measure_at=None, measure_fn=None):
        """Run ``n_steps`` interactions.

        If ``measure_at`` (a sorted sequence of step counts) and ``measure_fn``
        are given, ``measure_fn(self)`` is called after those steps and the
        results are returned as a list.
        """
        if self.shared_schedule:
            return self._run_shared(n_steps, measure_at, measure_fn)
        return self._run_independent(n_steps, measure_at, measure_fn)

    def _run_shared(self, n_steps, measure_at, measure_fn):
        out = []
        targets = list(measure_at or ())
        next_target = targets.pop(0) if targets else None
        N, P = self.N, self.P
        rng = self.rng
        # Draw the whole schedule up front: cheap, and keeps the inner loop
        # free of per-step generator calls.
        recv = rng.integers(N, size=n_steps)
        emit = rng.integers(N - 1, size=n_steps)
        emit += emit >= recv
        issue = rng.integers(P, size=n_steps)
        X = self.X
        for t in range(n_steps):
            self._interact(int(recv[t]), int(emit[t]), X[issue[t]])
            self.n_steps_taken += 1
            if next_target is not None and self.n_steps_taken >= next_target:
                out.append(measure_fn(self))
                next_target = targets.pop(0) if targets else None
        return out

    def _run_independent(self, n_steps, measure_at, measure_fn):
        """Per-run independent schedules: correct but ~3x slower (gathers)."""
        out = []
        targets = list(measure_at or ())
        next_target = targets.pop(0) if targets else None
        N, P, R = self.N, self.P, self.R
        rng = self.rng
        runs = np.arange(R)
        for _ in range(n_steps):
            recv = rng.integers(N, size=R)
            emit = rng.integers(N - 1, size=R)
            emit += emit >= recv
            x = self.X[rng.integers(P, size=R), runs]
            self._interact_gathered(recv, emit, runs, x)
            self.n_steps_taken += 1
            if next_target is not None and self.n_steps_taken >= next_target:
                out.append(measure_fn(self))
                next_target = targets.pop(0) if targets else None
        return out

    # -- the update itself -------------------------------------------------
    def _interact(self, r, e, x):
        """One interaction, shared indices: all accesses are views."""
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

        F_w, F_C, F_mu, F_V = modulation(h_w, h_mu, self.z_floor)

        # opinion sector
        w_r += ((F_w * sigma / gamma_C)[:, None]) * Cx
        a = F_C / (gamma_C * gamma_C)
        a = self._project_psd(a, xCx)
        C_r += a[:, None, None] * (Cx[:, :, None] * Cx[:, None, :])

        # trust sector
        self.mu[r, e] += (F_mu / gamma_V) * V
        np.maximum(V + (F_V / (gamma_V * gamma_V)) * V * V, self.v_floor, out=self.V[r, e])

    def _interact_gathered(self, recv, emit, runs, x):
        """One interaction with per-run indices (independent schedules)."""
        w_r = self.w[recv, runs]
        C_r = self.C[recv, runs]

        sigma = np.sign(np.einsum("rk,rk->r", self.w[emit, runs], x))
        sigma[sigma == 0] = 1.0

        Cx = np.einsum("rij,rj->ri", C_r, x)
        xCx = np.einsum("rk,rk->r", x, Cx)
        gamma_C = np.sqrt(1.0 + xCx)

        V = self.V[recv, emit, runs]
        gamma_V = np.sqrt(1.0 + V)

        h_w = np.einsum("rk,rk->r", w_r, x) * sigma / gamma_C + self.D[recv, emit, runs]
        h_mu = self.mu[recv, emit, runs] / gamma_V

        F_w, F_C, F_mu, F_V = modulation(h_w, h_mu, self.z_floor)

        a = self._project_psd(F_C / (gamma_C * gamma_C), xCx)
        self.w[recv, runs] = w_r + ((F_w * sigma / gamma_C)[:, None]) * Cx
        self.C[recv, runs] = C_r + a[:, None, None] * (Cx[:, :, None] * Cx[:, None, :])
        self.mu[recv, emit, runs] += (F_mu / gamma_V) * V
        self.V[recv, emit, runs] = np.maximum(
            V + (F_V / (gamma_V * gamma_V)) * V * V, self.v_floor
        )

    def _project_psd(self, a, xCx, margin=1e-9):
        """Keep the covariance update from pushing ``C`` out of PSD.

        ``C -> C + a (Cx)(Cx)^T`` stays positive semi-definite exactly when
        ``1 + a x.Cx >= 0``; the binding direction is ``x`` itself.  Since
        ``F_C`` is unbounded below near the dissonance corners, clip ``a`` to
        that boundary.  The draft is silent about this; the clip is rare (the
        counter below records how rare) and only ever *reduces* the magnitude of
        a step the exact algorithm would have taken.
        """
        lo = -(1.0 - margin) / np.maximum(xCx, 1e-300)
        n_bad = int(np.count_nonzero(a < lo))
        if n_bad:
            self.n_psd_clips += n_bad
            a = np.maximum(a, lo)
        return a
