"""Controlled small-C,V reduction of the entropic learning dynamics."""

from .fields import field_density
from .kernels import (
    K_C,
    affective_kernels,
    affective_kernels_mc,
    ideological_coefficients,
    ideological_coefficients_mc,
    ideological_moments,
)
from .microscopic import full_increment, leading_increment
from .modulation import F_C, F_V, F_mu, F_w, Phi, Z, phi

__all__ = [
    "F_C",
    "F_V",
    "F_mu",
    "F_w",
    "K_C",
    "Phi",
    "Z",
    "affective_kernels",
    "affective_kernels_mc",
    "field_density",
    "full_increment",
    "ideological_coefficients",
    "ideological_coefficients_mc",
    "ideological_moments",
    "leading_increment",
    "phi",
]
