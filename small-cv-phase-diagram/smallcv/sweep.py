"""Sweeps over the collective phase-diagram plane (d, f_d)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ModelConfig, SweepConfig
from .observables import ORDER_PARAM_NAMES, measure
from .society import SocietyBatch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


def _cache_payload(model, sweep_cfg, tag=""):
    sweep_dict = asdict(sweep_cfg)
    # Batch size controls checkpoint granularity only; it must not change the
    # scientific cache identity, otherwise a resumable run becomes tied to the
    # batch size used when it started.
    sweep_dict.pop("batch_size", None)
    return {"model": asdict(model), "sweep": sweep_dict, "tag": tag}


def cache_path(model, sweep_cfg, tag=""):
    payload = json.dumps(_cache_payload(model, sweep_cfg, tag), sort_keys=True)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    label = tag or f"{model.dynamics}_c{model.initial_c:g}_r{model.ratio_v_over_c:g}"
    return DATA_DIR / f"sweep_{label}_{sweep_cfg.n_fd}x{sweep_cfg.n_d}_{digest}.npz"


def _cache_stem(model, sweep_cfg, tag=""):
    return cache_path(model, sweep_cfg, tag).stem


def _checkpoint_path(model, sweep_cfg, tag, start, stop):
    return CHECKPOINT_DIR / f"{_cache_stem(model, sweep_cfg, tag)}__{start:06d}_{stop:06d}.npz"


def _matching_checkpoint(model, sweep_cfg, tag, start, stop):
    exact = _checkpoint_path(model, sweep_cfg, tag, start, stop)
    if exact.exists():
        return exact, start
    label = tag or f"{model.dynamics}_c{model.initial_c:g}_r{model.ratio_v_over_c:g}"
    pattern = f"sweep_{label}_{sweep_cfg.n_fd}x{sweep_cfg.n_d}_*__{start:06d}_{stop:06d}.npz"
    matches = sorted(CHECKPOINT_DIR.glob(pattern))
    if matches:
        return matches[-1], start

    cover_pattern = f"sweep_{label}_{sweep_cfg.n_fd}x{sweep_cfg.n_d}_*__*.npz"
    covering = []
    for candidate in CHECKPOINT_DIR.glob(cover_pattern):
        try:
            c_start, c_stop = candidate.stem.split("__")[-1].split("_")
            c_start, c_stop = int(c_start), int(c_stop)
        except ValueError:
            continue
        if c_start <= start and stop <= c_stop:
            covering.append((c_stop - c_start, c_start, candidate))
    if covering:
        _, c_start, candidate = sorted(covering)[-1]
        return candidate, c_start
    return exact, start


def _load_checkpoint(path, requested_start, requested_stop, checkpoint_start):
    with np.load(path) as z:
        piece = {k: z[k] for k in z.files}
    offset = requested_start - checkpoint_start
    length = requested_stop - requested_start
    if offset == 0 and next(iter(piece.values())).shape[0] == length:
        return piece
    return {k: v[offset:offset + length] for k, v in piece.items()}


def _run_batch(model, d_values, fd_values, seed):
    batch = SocietyBatch(
        n_agents=model.n_agents,
        n_dim=model.n_dim,
        n_issues=model.n_issues,
        d=d_values,
        f_d=fd_values,
        case=model.case,
        initial_c=model.initial_c,
        initial_v=model.initial_v,
        seed=seed,
        dynamics=model.dynamics,
        literal_draft_sign=model.literal_draft_sign,
    )
    batch.run(model.n_steps())
    out = measure(batch, class_indicator=model.class_indicator, literal_norm=model.literal_norm)
    out.update(batch.gamma_diagnostics())
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def sweep(model=None, sweep_cfg=None, tag="", use_cache=True, verbose=True):
    model = model or ModelConfig()
    sweep_cfg = sweep_cfg or SweepConfig()
    path = cache_path(model, sweep_cfg, tag)
    if use_cache and path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    d_axis, fd_axis = sweep_cfg.grids()
    D, F = np.meshgrid(d_axis, fd_axis)
    d_flat = np.repeat(D.ravel(), sweep_cfg.n_repeats)
    fd_flat = np.repeat(F.ravel(), sweep_cfg.n_repeats)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pieces = []
    for start in range(0, d_flat.size, sweep_cfg.batch_size):
        stop = min(start + sweep_cfg.batch_size, d_flat.size)
        ckpt, ckpt_start = _matching_checkpoint(model, sweep_cfg, tag, start, stop)
        if use_cache and ckpt.exists():
            if verbose:
                print(f"[sweep] loaded checkpoint {stop}/{d_flat.size} societies")
            pieces.append(_load_checkpoint(ckpt, start, stop, ckpt_start))
            continue

        if verbose:
            print(f"[sweep] running {start}-{stop}/{d_flat.size} societies")
        piece = _run_batch(model, d_flat[start:stop], fd_flat[start:stop], sweep_cfg.seed + start)
        np.savez_compressed(ckpt, **piece)
        pieces.append(piece)
        if verbose:
            print(f"[sweep] checkpointed {stop}/{d_flat.size} societies")

    shape = (sweep_cfg.n_fd, sweep_cfg.n_d, sweep_cfg.n_repeats)
    result = {"d": d_axis, "fd": fd_axis}
    for name in ORDER_PARAM_NAMES:
        values = np.concatenate([p[name] for p in pieces]).reshape(shape)
        result[name] = values.mean(axis=2)
        result[f"{name}_std"] = values.std(axis=2)
    for name in ("max_gamma_C_minus_1", "max_gamma_V_minus_1"):
        values = np.concatenate([p[name].reshape(-1) for p in pieces]).reshape(shape)
        result[name] = values.max(axis=2)

    np.savez_compressed(path, **result)
    if verbose:
        print(f"[sweep] cached -> {path}")
    return result
