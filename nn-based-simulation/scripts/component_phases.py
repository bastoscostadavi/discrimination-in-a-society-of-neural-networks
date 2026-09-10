#!/usr/bin/env python3
"""Two slices of the general group-bias field, side by side.

A class-dependent shift of the opinion field decomposes into four orthogonal
parts (see the paper's appendix on the prejudice field),

    D[R, E] = a + b*kappa_R*kappa_E + c*kappa_R + d*kappa_E

and the main text varies only ``b``.  This figure shows the ``(a, f_a)`` and
``(c, f_c)`` slices using sweeps from their respective projects.

Each panel keeps the composite its own project defines, because the two planes
are not describable by one set of order parameters.  ``a`` names no class at all
and cannot split anything: what it varies
is how far the whole population is driven into universal trust or universal
distrust, which is ``T_mu``, signed, and which the paper's ``R_muc`` does not
see.  ``c`` singles out a class as the credulous one, so its channel is the
credulity split ``R_cred`` against how differently the two classes cohere.
Compositing both on the paper's triple makes them look empty --
not because nothing happens in them, but because the instrument is built for a
different phenomenon.

``a`` and ``c`` were run as five horizontal strips of forty rows each, so their
planes are stitched back together here.  The strength axes differ in extent:
``a`` runs over [-1, 1], while for ``c`` negating the field is exactly
the relabelling ``A <-> B``, which maps the ensemble to itself, so only the
positive half was swept.

Writes ``component_phases``.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt

from _cli import setup  # noqa: E402

from ednna.plotting import save, text_width  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _sibling(name):
    """Import a sibling project's plotting module without disturbing ours."""
    path = str(ROOT / name)
    if path not in sys.path:
        sys.path.insert(0, path)


_sibling("uniform-credulity")
_sibling("credulity-asymmetry")
from credulity.plotting import credulity_composite  # noqa: E402
from credfield.plotting import channel_composite  # noqa: E402

#: symbol -> (project, strip glob, strength-axis key)
STRIPS = {
    "a": ("uniform-credulity", "sweep_a_P5_N40_200x40_*.npz", "a"),
    "c": ("credulity-asymmetry", "sweep_b_P5_strip*of5_200x40_*.npz", "s"),
}

def _stitch(project, pattern, x_key):
    """Reassemble a plane from the horizontal strips it was run in.

    Strips carry no index in their filename, only a hash, so they are ordered by
    the f range they cover rather than by name.
    """
    paths = sorted((ROOT / project / "data").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no strips matching {pattern} in {project}/data")
    pieces = sorted((dict(np.load(p)) for p in paths),
                    key=lambda z: float(z["f"].min()))
    x = pieces[0][x_key]
    for z in pieces[1:]:
        if not np.allclose(z[x_key], x):
            raise ValueError(f"{project}: strips disagree on the {x_key} axis")
    keys = set(pieces[0]) - {x_key, "f"}
    out = {"x": x, "f": np.concatenate([z["f"] for z in pieces])}
    out.update({k: np.concatenate([z[k] for z in pieces], axis=0) for k in keys})
    return out


def _atomization(z):
    """``(B_eta^AA - B_eta^BB)/2``, the green channel of the ``c`` composite."""
    return 0.5 * (z["B_eta_A"] - z["B_eta_B"])


def figure(preset, style, use_cache=True, name="component_phases"):
    planes = {s: _stitch(*STRIPS[s]) for s in STRIPS}

    composites = {
        "a": lambda z: credulity_composite(z["T_mu"], z["R_wmu"]),
        "c": lambda z: channel_composite(z["R_cred"], _atomization(z), z["R_wmu"]),
    }

    order = ("a", "c")
    W = text_width()
    fig, axes = plt.subplots(1, len(order), figsize=(W, W / len(order) * 1.12))
    for ax, sym in zip(axes, order):
        z = planes[sym]
        x, f = z["x"], z["f"]
        ax.imshow(composites[sym](z), origin="lower",
                  extent=[x[0], x[-1], f[0], f[-1]], aspect="auto")
        ax.set_box_aspect(1)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(f[0], f[-1])
        ax.set_xlabel(f"${sym}$", labelpad=1)
        ax.set_ylabel(rf"$f_{sym}$", labelpad=1)
        ax.set_xticks(np.linspace(x[0], x[-1], 5))
        ax.set_yticks(np.linspace(f[0], f[-1], 5))
    fig.tight_layout(pad=0.4, w_pad=1.3)
    return save(fig, name, style)


def main():
    args, preset = setup(__doc__)
    figure(preset, args.style, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
