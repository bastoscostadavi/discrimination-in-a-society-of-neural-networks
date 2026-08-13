#!/usr/bin/env python3
"""Put each generated figure next to the corresponding page of the source draft.

The source draft is our reference for what the model does, but it states none of
the simulation parameters, so agreement has to be judged rather than asserted.
This builds one sheet, ``figures/draft_comparison.pdf``, with our figure on the
left and the cropped region of the draft's page on the right, so that the
comparison is visible.

Requires ``pdftoppm`` (poppler) and the source PDF in the repository root.

    python scripts/draft_comparison.py --style paper
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from _cli import setup  # noqa: E402

from ednna.plotting import FIGURE_DIR  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
DRAFT = next(REPO.glob("Discrimination*.pdf"), None)

#: our figure -> (page in the draft, y-crop as a fraction of page height, note)
#: Page numbers are 1-based positions in the PDF file.
PAIRS = [
    ("modulation_contours", 13, (0.05, 0.45),
     "draft Fig. 3: contour maps of the four modulation functions"),
    ("learning_flows", 14, (0.05, 0.30),
     "draft Fig. 4: learning flow at D = -d, 0, +d"),
    ("correlation_maps", 16, (0.05, 0.45),
     "draft Fig. 5: the three correlations, two agenda sizes"),
    ("frustration_maps", 17, (0.05, 0.60),
     "draft Fig. 6: balance maps, two agenda sizes"),
    ("phase_diagram", 5, (0.06, 0.40),
     "draft Fig. 1: composite phase diagram"),
    ("agenda_trajectories", 18, (0.05, 0.38),
     "draft Fig. 7: balance trajectories vs agenda complexity"),
]


def render_pdf(pdf, prefix, out_dir, page=1, dpi=150):
    """Rasterise one page of a PDF and return the image path."""
    stem = out_dir / prefix
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", "-f", str(page), "-l", str(page),
         str(pdf), str(stem)],
        check=True, capture_output=True,
    )
    matches = sorted(out_dir.glob(f"{prefix}-*.png"))
    if not matches:
        raise FileNotFoundError(f"pdftoppm produced nothing for {pdf} page {page}")
    return matches[0]


def crop(img, y_frac):
    h = img.shape[0]
    y0, y1 = int(h * y_frac[0]), int(h * y_frac[1])
    return img[y0:y1]


def main():
    args, _ = setup(__doc__)
    if DRAFT is None:
        sys.exit("source draft PDF not found in the repository root")
    if shutil.which("pdftoppm") is None:
        sys.exit("pdftoppm not found; install poppler")

    fig_dir = FIGURE_DIR / args.style
    out = FIGURE_DIR / "draft_comparison.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with PdfPages(out) as pdf:
            for name, page, y_frac, note in PAIRS:
                ours = fig_dir / f"{name}.pdf"
                if not ours.exists():
                    print(f"  skipping {name}: not generated yet")
                    continue
                ours_img = mpimg.imread(render_pdf(ours, f"ours_{name}", tmp))
                theirs = crop(mpimg.imread(render_pdf(DRAFT, f"p{page}", tmp, page=page)), y_frac)
                fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.5))
                for ax, img, title in (
                    (axes[0], ours_img, f"this simulation: {name}"),
                    (axes[1], theirs, note),
                ):
                    ax.imshow(np.asarray(img))
                    ax.set_title(title, fontsize=9)
                    ax.axis("off")
                fig.tight_layout(pad=0.6)
                pdf.savefig(fig)
                plt.close(fig)
                print(f"  {name} vs draft page {page}")
    print(f"[comparison] {out}")


if __name__ == "__main__":
    main()
