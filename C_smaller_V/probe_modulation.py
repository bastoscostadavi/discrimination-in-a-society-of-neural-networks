#!/usr/bin/env python3
"""Numerical checks for the C << V fast-affective asymptotics."""

from __future__ import annotations

import math
from pathlib import Path


SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-x / SQRT_2)


def Z(H: float, mu: float) -> float:
    return Phi(H) + Phi(mu) - 2.0 * Phi(H) * Phi(mu)


def modulation(H: float, mu: float) -> tuple[float, float]:
    evidence = Z(H, mu)
    Fw = (1.0 - 2.0 * Phi(mu)) * phi(H) / evidence
    Fmu = (1.0 - 2.0 * Phi(H)) * phi(mu) / evidence
    return Fw, Fmu


def saturated_M(H: float) -> float:
    if H > 0:
        return phi(H) / Phi(H)
    if H < 0:
        return -phi(H) / (1.0 - Phi(H))
    return 0.0


def main() -> None:
    out = Path(__file__).with_name("modulation_probe.csv")
    H_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
    mu_values = [-6, -3, 0, 3, 6]
    lines = ["H,mu,F_w,F_mu,sign_F_mu,saturated_M"]
    for H in H_values:
        for mu in mu_values:
            Fw, Fmu = modulation(H, mu)
            if Fmu > 1e-12:
                sign = 1
            elif Fmu < -1e-12:
                sign = -1
            else:
                sign = 0
            lines.append(f"{H:g},{mu:g},{Fw:.12g},{Fmu:.12g},{sign},{saturated_M(H):.12g}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()

