#!/usr/bin/env python3
"""Generate C << V phase graphs with only NumPy and PIL."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def erf_approx(x: np.ndarray) -> np.ndarray:
    """Vectorized Abramowitz-Stegun erf approximation."""

    sign = np.sign(x)
    ax = np.abs(x)
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


def case6_matrix(d: float) -> np.ndarray:
    return d * np.array([[1.0, -1.0], [-1.0, 1.0]])


def frozen_observables(
    d_values: np.ndarray,
    fd_values: np.ndarray,
    *,
    n_repeats: int = 12,
    n_agents: int = 40,
    n_dim: int = 30,
    n_issues: int = 5,
    seed: int = 20260814,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    R_muc = np.zeros((len(fd_values), len(d_values)))
    R_cw = np.zeros_like(R_muc)
    locked = np.zeros_like(R_muc)

    class_of = np.zeros(n_agents, dtype=int)
    class_of[n_agents // 2 :] = 1
    kappa = np.where(class_of == 0, 1.0, -1.0)
    G = np.outer(kappa, kappa)
    iu = np.triu_indices(n_agents, 1)
    denom = n_agents * (n_agents - 1)

    for _ in range(n_repeats):
        w = rng.normal(size=(n_agents, n_dim))
        w_unit = w / np.linalg.norm(w, axis=1, keepdims=True)
        rho = w_unit @ w_unit.T
        X = rng.normal(size=(n_issues, n_dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        wx = w @ X.T
        sigma = np.sign(wx)
        sigma[sigma == 0] = 1.0
        h0 = np.einsum("rp,ep->rep", wx, sigma)

        for i_fd, fd in enumerate(fd_values):
            discriminates = rng.random(n_agents) < fd
            for i_d, d in enumerate(d_values):
                Dmat = case6_matrix(float(d))
                D = Dmat[np.ix_(class_of, class_of)] * discriminates[:, None]
                H = h0 + D[:, :, None]
                score_pair = erf_approx(H / math.sqrt(2.0)).mean(axis=2)
                eta = np.sign(score_pair)
                np.fill_diagonal(eta, 1.0)
                S = eta + eta.T
                R_muc[i_fd, i_d] += (S[iu] * G[iu]).sum() / denom
                R_cw[i_fd, i_d] += (rho[iu] * G[iu]).sum() * 2.0 / denom
                locked[i_fd, i_d] += (eta[iu] == np.sign(G[iu])).mean()

    return {
        "R_muc": R_muc / n_repeats,
        "R_cw": R_cw / n_repeats,
        "locked": locked / n_repeats,
    }


def blend(c0: tuple[int, int, int], c1: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    a = np.array(c0, dtype=float)
    b = np.array(c1, dtype=float)
    return (a + (b - a) * t[..., None]).clip(0, 255).astype(np.uint8)


def cmap_rdbu(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    blue = np.array([49, 94, 150])
    white = np.array([246, 246, 246])
    red = np.array([178, 45, 38])
    out = np.empty(values.shape + (3,), dtype=np.uint8)
    low = x < 0.5
    out[low] = blend(tuple(blue), tuple(white), x[low] * 2)
    out[~low] = blend(tuple(white), tuple(red), (x[~low] - 0.5) * 2)
    return out


def cmap_viridis(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    x = np.clip((values - vmin) / (vmax - vmin), 0, 1)
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    pos = x * (len(stops) - 1)
    idx = np.minimum(pos.astype(int), len(stops) - 2)
    frac = pos - idx
    out = np.zeros(values.shape + (3,), dtype=np.uint8)
    for i in range(len(stops) - 1):
        mask = idx == i
        if np.any(mask):
            out[mask] = blend(stops[i], stops[i + 1], frac[mask])
    return out


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_heatmap_panel(draw: ImageDraw.ImageDraw, img: Image.Image, values: np.ndarray, box, title, vmin, vmax, cmap):
    x0, y0, x1, y1 = box
    heat = cmap(values, vmin, vmax)
    heat = np.flipud(heat)
    hm = Image.fromarray(heat, "RGB").resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
    img.paste(hm, (x0, y0))
    draw.rectangle(box, outline=(40, 40, 40), width=1)
    draw.text((x0, y0 - 38), title, fill=(20, 20, 20), font=font(18))
    draw.text((x0 + 95, y1 + 12), "d", fill=(20, 20, 20), font=font(16))
    draw.text((x0 - 30, y0 + 100), "f_d", fill=(20, 20, 20), font=font(16))
    draw.text((x0 - 7, y1 + 4), "-1", fill=(20, 20, 20), font=font(12))
    draw.text((x0 + (x1 - x0) // 2 - 5, y1 + 4), "0", fill=(20, 20, 20), font=font(12))
    draw.text((x1 - 10, y1 + 4), "1", fill=(20, 20, 20), font=font(12))
    draw.text((x0 - 24, y1 - 8), "0", fill=(20, 20, 20), font=font(12))
    draw.text((x0 - 24, y0 - 4), "1", fill=(20, 20, 20), font=font(12))
    x_mid = x0 + (x1 - x0) // 2
    draw.line((x_mid, y0, x_mid, y1), fill=(0, 0, 0), width=1)

    cbx = x1 + 10
    grad_vals = np.linspace(vmax, vmin, y1 - y0)[:, None]
    grad = cmap(grad_vals, vmin, vmax).repeat(14, axis=1)
    img.paste(Image.fromarray(grad, "RGB"), (cbx, y0))
    draw.rectangle((cbx, y0, cbx + 14, y1), outline=(50, 50, 50), width=1)
    draw.text((cbx + 18, y0 - 3), f"{vmax:g}", fill=(20, 20, 20), font=font(11))
    draw.text((cbx + 18, y1 - 10), f"{vmin:g}", fill=(20, 20, 20), font=font(11))


def save_phase(data: dict[str, np.ndarray], out: Path) -> None:
    img = Image.new("RGB", (1320, 470), "white")
    draw = ImageDraw.Draw(img)
    draw.text((32, 20), "Leading-order C << V phase prediction", fill=(10, 10, 10), font=font(24))
    draw.text((32, 52), "Fast trust orders by class while frozen opinions remain nearly uncorrelated.", fill=(60, 60, 60), font=font(15))
    boxes = [(60, 125, 355, 390), (500, 125, 795, 390), (940, 125, 1235, 390)]
    draw_heatmap_panel(draw, img, data["R_muc"], boxes[0], "fast trust-class R_mu,c", -1, 1, cmap_rdbu)
    draw_heatmap_panel(draw, img, data["R_cw"], boxes[1], "frozen opinion-class R_c,w", -0.2, 0.2, cmap_rdbu)
    draw_heatmap_panel(draw, img, data["locked"], boxes[2], "class-locked directed trust", 0, 1, cmap_viridis)
    img.save(out)


def save_timescale(out: Path) -> None:
    img = Image.new("RGB", (900, 460), "white")
    draw = ImageDraw.Draw(img)
    draw.text((38, 22), "C/V controls the delay from phase IV to phase III", fill=(10, 10, 10), font=font(24))
    x0, y0, x1, y1 = 90, 85, 835, 385
    draw.line((x0, y1, x1, y1), fill=(20, 20, 20), width=2)
    draw.line((x0, y0, x0, y1), fill=(20, 20, 20), width=2)
    draw.text((360, 410), "time on fast trust scale", fill=(20, 20, 20), font=font(15))
    draw.text((12, 215), "order", fill=(20, 20, 20), font=font(15))

    t = np.linspace(0, 6, 240)
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44)]
    eps_values = [1.0, 0.1, 0.01]

    def xy(tt, yy):
        return x0 + tt / 6 * (x1 - x0), y1 - yy / 0.9 * (y1 - y0)

    Rmuc = 0.85 * (1 - np.exp(-t))
    pts = [xy(float(a), float(b)) for a, b in zip(t, Rmuc)]
    draw.line(pts, fill=(0, 0, 0), width=4)
    draw.text((650, 115), "R_mu,c fast", fill=(0, 0, 0), font=font(15))

    for eps, col in zip(eps_values, colors):
        Rcw = 0.85 * (1 - np.exp(-eps * t))
        pts = [xy(float(a), float(b)) for a, b in zip(t, Rcw)]
        draw.line(pts, fill=col, width=3)
        draw.text((610, 255 + 28 * eps_values.index(eps)), f"R_c,w   C/V={eps:g}", fill=col, font=font(15))
    img.save(out)


def main() -> None:
    root = Path(__file__).resolve().parent
    d_values = np.linspace(-1.0, 1.0, 81)
    fd_values = np.linspace(0.0, 1.0, 61)
    data = frozen_observables(d_values, fd_values)
    phase = root / "c_smaller_v_phase_prediction.png"
    timescale = root / "c_smaller_v_timescale_delay.png"
    save_phase(data, phase)
    save_timescale(timescale)
    print(phase)
    print(timescale)


if __name__ == "__main__":
    main()

