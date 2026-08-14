"""Full and small-C,V microscopic EDNNA society dynamics."""

from __future__ import annotations

import numpy as np

from .discrimination import field_matrix
from .modulation import Z_FLOOR, modulation

V_FLOOR = 1e-12
DYNAMICS = ("small_cv", "full")


class SocietyBatch:
    """Vectorized batch of societies sharing an interaction schedule."""

    def __init__(
        self,
        n_agents=40,
        n_dim=30,
        n_issues=5,
        d=0.0,
        f_d=0.0,
        case=6,
        initial_c=0.05,
        initial_v=0.05,
        seed=0,
        dynamics="small_cv",
        literal_draft_sign=False,
        z_floor=Z_FLOOR,
        v_floor=V_FLOOR,
    ):
        if dynamics not in DYNAMICS:
            raise ValueError(f"dynamics must be one of {DYNAMICS}, got {dynamics!r}")
        if initial_c <= 0.0 or initial_v <= 0.0:
            raise ValueError("initial_c and initial_v must be positive")

        d = np.atleast_1d(np.asarray(d, dtype=float))
        f_d = np.atleast_1d(np.asarray(f_d, dtype=float))
        R = max(d.size, f_d.size)
        d = np.repeat(d, R) if d.size == 1 else d
        f_d = np.repeat(f_d, R) if f_d.size == 1 else f_d
        if d.size != R or f_d.size != R:
            raise ValueError("d and f_d must have the same length or be scalars")

        self.N = int(n_agents)
        self.K = int(n_dim)
        self.P = int(n_issues)
        self.R = int(R)
        self.d = d
        self.f_d = f_d
        self.case = case
        self.initial_c = float(initial_c)
        self.initial_v = float(initial_v)
        self.dynamics = dynamics
        self.z_floor = z_floor
        self.v_floor = v_floor
        self.n_steps_taken = 0
        self.n_psd_clips = 0
        self.max_gamma_C_minus_1 = np.zeros(R)
        self.max_gamma_V_minus_1 = np.zeros(R)

        rng = np.random.default_rng(seed)
        self.rng = rng
        N, K, P = self.N, self.K, self.P

        self.class_of = np.zeros(N, dtype=np.int8)
        self.class_of[N // 2 :] = 1
        self.kappa = np.where(self.class_of == 0, 1.0, -1.0)

        self.w = rng.normal(size=(N, R, K))
        self.C = np.zeros((N, R, K, K))
        self.C[:, :] = self.initial_c * np.eye(K)
        self.mu = rng.uniform(-1.0, 1.0, size=(N, N, R))
        self.V = np.full((N, N, R), self.initial_v)

        X = rng.normal(size=(P, R, K))
        X /= np.linalg.norm(X, axis=2, keepdims=True)
        self.X = X

        self.discriminates = rng.random((N, R)) < f_d[None, :]
        self.D = np.zeros((N, N, R))
        for run in range(R):
            M = field_matrix(d[run], case=case, literal_draft=literal_draft_sign)
            self.D[:, :, run] = M[np.ix_(self.class_of, self.class_of)] * self.discriminates[:, run][:, None]

    @property
    def n_interactions_per_channel(self):
        return self.n_steps_taken / (self.N * (self.N - 1))

    def gamma_diagnostics(self):
        return {
            "max_gamma_C_minus_1": self.max_gamma_C_minus_1.copy(),
            "max_gamma_V_minus_1": self.max_gamma_V_minus_1.copy(),
        }

    def step(self):
        N = self.N
        r = int(self.rng.integers(N))
        e = int(self.rng.integers(N - 1))
        if e >= r:
            e += 1
        p = int(self.rng.integers(self.P))
        self.interact(r, e, self.X[p])
        self.n_steps_taken += 1

    def run(self, n_steps, measure_at=None, measure_fn=None):
        out = []
        targets = list(measure_at or ())
        next_target = targets.pop(0) if targets else None
        recv = self.rng.integers(self.N, size=n_steps)
        emit = self.rng.integers(self.N - 1, size=n_steps)
        emit += emit >= recv
        issue = self.rng.integers(self.P, size=n_steps)
        for t in range(n_steps):
            self.interact(int(recv[t]), int(emit[t]), self.X[issue[t]])
            self.n_steps_taken += 1
            if next_target is not None and self.n_steps_taken >= next_target:
                out.append(measure_fn(self))
                next_target = targets.pop(0) if targets else None
        return out

    def interact(self, r, e, x):
        w_r = self.w[r]
        C_r = self.C[r]
        sigma = np.sign(np.einsum("rk,rk->r", self.w[e], x))
        sigma[sigma == 0] = 1.0

        Cx = np.einsum("rij,rj->ri", C_r, x)
        xCx = np.einsum("rk,rk->r", x, Cx)
        gamma_C_true = np.sqrt(1.0 + xCx)
        V = self.V[r, e]
        gamma_V_true = np.sqrt(1.0 + V)
        self._record_gamma_deviation(gamma_C_true, gamma_V_true)

        if self.dynamics == "small_cv":
            gamma_C = 1.0
            gamma_V = 1.0
        else:
            gamma_C = gamma_C_true
            gamma_V = gamma_V_true

        h_w = np.einsum("rk,rk->r", w_r, x) * sigma / gamma_C + self.D[r, e]
        h_mu = self.mu[r, e] / gamma_V
        F_w, F_C, F_mu, F_V = modulation(h_w, h_mu, self.z_floor)

        w_r += ((F_w * sigma / gamma_C)[:, None]) * Cx
        a = self._project_psd(F_C / (gamma_C * gamma_C), xCx)
        C_r += a[:, None, None] * (Cx[:, :, None] * Cx[:, None, :])
        self.mu[r, e] += (F_mu / gamma_V) * V
        np.maximum(V + (F_V / (gamma_V * gamma_V)) * V * V, self.v_floor, out=self.V[r, e])

    def _record_gamma_deviation(self, gamma_C, gamma_V):
        np.maximum(self.max_gamma_C_minus_1, np.abs(gamma_C - 1.0), out=self.max_gamma_C_minus_1)
        np.maximum(self.max_gamma_V_minus_1, np.abs(gamma_V - 1.0), out=self.max_gamma_V_minus_1)

    def _project_psd(self, a, xCx, margin=1e-9):
        lo = -(1.0 - margin) / np.maximum(xCx, 1e-300)
        n_bad = int(np.count_nonzero(a < lo))
        if n_bad:
            self.n_psd_clips += n_bad
            a = np.maximum(a, lo)
        return a
