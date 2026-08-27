#!/usr/bin/env python3
"""Print every threshold and regime number this directory's README quotes.

    python scripts/thresholds.py                       # the b plane alone
    python scripts/thresholds.py --also ../directional-prejudice/data/sweep_c_*.npz

Reads cached sweeps only -- it never simulates, so it costs a second.  The
``--also`` planes are compared under **one** definition, which is the point:
this repository's directories otherwise use different conventions, and the
difference between conventions is larger than the difference between the planes.

Any ``.npz`` written by a ``credfield``/``dirfield``-style sweep loads, so this
runs on a sibling's plane without copying its code.  The channel is inferred from
which of the four is actually responding, so no flag says whether a file is a
``b`` plane or a ``c`` plane.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from _cli import ROOT  # noqa: F401  (puts the package on sys.path)

from credfield.config import default_s_range, get_preset  # noqa: E402
from credfield.order_params import CHANNEL_NAMES  # noqa: E402
from credfield.thresholds import (  # noqa: E402
    WIDE_TRANSITION, compare, crossing, profile, regime_table, row_saturation,
    smooth, summarise, transition_width,
)

FIXED_LEVELS = (0.5, 0.6, 0.7)


def responding_channel(data):
    """Which trust channel this plane drives, read off the plane itself.

    The largest saturated magnitude wins.  Inferring it rather than being told
    means a sibling's cache needs no metadata and cannot be mislabelled.
    """
    scores = {k: float(np.abs(np.asarray(data[k])[-10:, -10:]).mean())
              for k in CHANNEL_NAMES if k in data}
    return max(scores, key=scores.get)


def load_striped_b():
    """This directory's own plane, reassembled from its strips."""
    import cred_asymmetry as ca
    preset = get_preset("full")
    cfg = preset.sweep.with_(s_range=default_s_range("b"))
    return ca._run_striped(preset.model, cfg, 5, use_cache=True)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def report_one(label, data, channel):
    s = np.asarray(data["s"])
    print(f"\n=== {label}: channel {channel}, grid {data[channel].shape} ===")

    ch_p = profile(data, channel)
    at_p = 0.5 * (profile(data, "B_eta_A") - profile(data, "B_eta_B"))
    rw_p = profile(data, "R_wmu")
    sat_ch, sat_at = smooth(ch_p)[-20:].mean(), smooth(at_p)[-20:].mean()
    print(f"  pooled over f >= 0.95: saturated channel {sat_ch:.3f}, "
          f"saturated atomization {sat_at:.3f}")
    # Both the smoothed and the raw crossing are printed, because the difference
    # between them is the sensitivity of the extraction -- and on these planes it
    # is LARGER than the difference between the planes, so quoting one number to
    # three decimals would claim a precision the method does not have.
    for name, y, lev in (("channel half-saturation", ch_p, None),
                         ("atomization half-saturation", at_p, None),
                         ("R_wmu falls below 0.1", -rw_p, -0.1)):
        ys = smooth(y)
        target = lev if lev is not None else ys[-20:].mean() / 2
        raw_t = lev if lev is not None else y[-20:].mean() / 2
        sm, raw = crossing(s, ys, target), crossing(s, y, raw_t)
        print(f"  {name:<30}: {sm:.3f}   (unsmoothed {raw:.3f}, "
              f"so +-{abs(sm-raw):.3f} to that choice)")

    # The transition width, because every failure mode of both threshold
    # definitions is a function of it, and a verdict without it is unquotable.
    ch = np.asarray(data[channel])
    widths = [transition_width(s, ch[i]) for i in range(ch.shape[0])
              if np.isfinite(row_saturation(ch[i], smooth_width=7))
              and row_saturation(ch[i], smooth_width=7) >= 0.4]
    widths = np.asarray([w for w in widths if np.isfinite(w)])
    if widths.size:
        wide = float((widths > WIDE_TRANSITION).mean())
        print(f"  transition width (25-75%)     : median {np.median(widths):.3f} "
              f"over {widths.size} rows, range [{widths.min():.3f}, "
              f"{widths.max():.3f}]")
        print(f"  {'':<30}  {100*wide:.0f}% of rows wider than {WIDE_TRANSITION}"
              f" -> locus verdict {'WEAK, quote the width' if wide > 0.5 else 'in the safe regime'}")

    print("  channel up the prevalence axis at full strength: ", end="")
    f = np.asarray(data["f"])
    ms = s >= 0.95
    print(", ".join(f"{np.asarray(data[channel])[int(np.abs(f-t).argmin()), ms].mean():.2f}"
                    for t in (0.2, 0.4, 0.6, 0.8, 1.0)))

    print(f"\n  {'R_muc by regime':<34}{'pixels':>8}{'mean':>8}{'sd':>8}{'max|.|':>8}")
    for lo, hi, n, mu, sd, mx in regime_table(data, channel):
        name = (f"below the transition (<= {hi})" if lo is None else
                f"saturated (>= {lo})" if hi is None else
                f"transition band ({lo} - {hi})")
        print(f"  {name:<34}{n:>8}{mu:>8.3f}{sd:>8.3f}{mx:>8.3f}")

    rm = np.asarray(data["R_muc"])
    i, j = np.unravel_index(np.abs(rm).argmax(), rm.shape)
    print(f"  worst pixel: R_muc = {rm[i,j]:+.3f} at strength {s[j]:.3f}, "
          f"f = {f[i]:.3f}, where the channel reads {data[channel][i,j]:.3f}")


def report_definitions(planes):
    """The same planes under both definitions, side by side."""
    print("\n\n=== what the threshold is a threshold IN ===")
    print("Each block fixes one definition and applies it to every plane.  The "
          "answer\ndepends on the definition, so the definition is printed with "
          "it.\n")
    for level in ("relative",) + FIXED_LEVELS:
        name = ("half of each row's own saturation" if level == "relative"
                else f"fixed absolute level {level}")
        print(f"  -- {name} --")
        print(f"     {'plane':<16}{'rows':>6}{'s_c':>8}{'rel sd':>8} | "
              f"{'f*s_c':>8}{'rel sd':>8}   conserved")
        for label, out in compare(planes, level=level).items():
            if "strength" not in out:
                print(f"     {label:<16}{out['n_rows']:>6}   "
                      f"{out.get('reason', 'no transition in range')}")
                continue
            # The numbers are printed either way; only the verdict is withheld,
            # and the reason says which gate or which principle withheld it.
            verdict = out["conserved"] or (
                f"[withheld: would have said {out['would_have_said']}]"
                if "would_have_said" in out else "[withheld]")
            print(f"     {label:<16}{out['n_rows']:>6}{out['strength']:>8.3f}"
                  f"{out['strength_spread']:>8.3f} | {out['product']:>8.3f}"
                  f"{out['product_spread']:>8.3f}   {verdict}"
                  + (f"  (by {out['margin']:.1f}x)" if out["conserved"] else ""))
        if level != "relative":
            print("       ^ numbers describe the strength/prevalence trade-off; "
                  "the locus verdict is\n         withheld because a fixed level "
                  "cannot locate a transition (see below).")
        print()
    print("  Only the relative definition can locate a transition.  A fixed "
          "absolute level\n  manufactures a prevalence dependence whenever "
          "saturation varies with prevalence:\n  on a synthetic plane that is "
          "vertical by construction it inverts the verdict\n  from about "
          "transition width 0.06 upwards, with both the span and the margin\n"
          "  gate passing comfortably.  Its numbers still answer a real question "
          "-- how much\n  bias buys a stated degree of order -- so they are "
          "printed and the verdict is not.\n\n  The relative definition is far "
          "more robust but not immune: it survives every\n  width up to 0.40 on "
          "the planes built here, and has been seen to invert near\n  0.20 on an "
          "independently built one.  So quote the transition width beside any\n"
          "  locus verdict (tests/test_thresholds.py).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--also", nargs="*", default=[],
                    help="further cached sweeps (.npz, globs allowed) to compare "
                         "under the same definition")
    args = ap.parse_args()

    planes = {}
    b = load_striped_b()
    planes["b (here)"] = (b, responding_channel(b))
    for pattern in args.also:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            d = load_npz(path)
            planes[Path(path).stem[:16]] = (d, responding_channel(d))

    for label, (data, channel) in planes.items():
        report_one(label, data, channel)
    report_definitions(planes)


if __name__ == "__main__":
    main()
