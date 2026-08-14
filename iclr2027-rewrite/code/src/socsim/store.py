"""Where results go, and why the layout is long rather than gridded.

Results are stored one row per *society* --- ``(point, disorder replicate, init
replicate)`` --- rather than as a ``(n_fd, n_d, n_seeds)`` block.  Four things
follow, all of which a multi-hour background campaign needs:

* **Adaptive refinement appends.**  Extra points near a boundary are new rows,
  not a reshaped array.
* **An interrupted run is simply missing rows**, so resuming is a set
  difference on ``(point_id, replicate_dis, replicate_init)`` and costs nothing.
* **Replicate counts may vary by point**, which is what putting more seeds near
  a boundary means.
* **Nothing is averaged on write.**  The reference implementation collapsed its
  repeats with ``.mean(axis=2)`` before saving, so no uncertainty could ever be
  recovered from a stored result --- the single most costly design decision in
  it, since the compute had already been spent.

Workers return about twenty floats per society and never any state, so memory is
bounded by construction: the largest run in the campaign is a few megabytes.

Shards are written as each chunk completes and merged afterwards.  A killed run
therefore loses at most one chunk, and a merge refuses to combine shards whose
configuration or code fingerprint differ, so two incompatible runs cannot be
silently averaged together.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .observables import OBS_NAMES

__all__ = [
    "SCHEMA_VERSION",
    "code_fingerprint",
    "write_shard",
    "merge",
    "load",
    "to_grid",
    "pending",
]

SCHEMA_VERSION = 1

_STR = "U64"


#: Modules whose contents can change a society's trajectory or its measured
#: values.  Everything else in the package -- the runner, the store, plotting,
#: the campaign table, the classifier, the analysis helpers -- decides *which*
#: societies are simulated and how results are presented, never what a given
#: society does.
PHYSICS_MODULES = (
    "modulation.py",
    "society.py",
    "discrimination.py",
    "observables.py",
    "seeds.py",
    "config.py",
)


def code_fingerprint():
    """A hash of the modules that determine the physics, stored with every result.

    Cheap insurance: if a campaign spans a change to the dynamics or the
    observables, the merge fails loudly instead of silently pooling societies
    simulated under different rules.

    It deliberately covers only :data:`PHYSICS_MODULES`.  Hashing the whole
    package was the first attempt and it was too strict to be useful: fixing a
    bug in how shard *files are named* changed the fingerprint and made a
    completed run unmergeable, even though nothing about any society had
    changed.  A guard that fires on edits which provably cannot matter is one
    that gets disabled, so it is scoped to the edits that can.
    """
    here = Path(__file__).resolve().parent
    h = hashlib.blake2b(digest_size=8)
    for name in PHYSICS_MODULES:
        p = here / name
        h.update(name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _rows_to_arrays(rows, obs, params):
    out = {
        "point_id": np.array([r["point_id"] for r in rows], dtype=_STR),
        "replicate_dis": np.array([r["replicate_dis"] for r in rows], dtype=np.int32),
        "replicate_init": np.array([r["replicate_init"] for r in rows], dtype=np.int32),
    }
    for name in params:
        out[name] = np.array([r[name] for r in rows], dtype=np.float64)
    out["obs_names"] = np.array(list(obs.keys()), dtype=_STR)
    out["obs"] = np.stack([np.asarray(obs[k], dtype=np.float64) for k in obs], axis=1)
    return out


def write_shard(path, spec, rows, obs, params, diagnostics=None):
    """Write one chunk's results.

    ``rows`` is a list of key dictionaries, ``obs`` maps observable name to an
    array of length ``len(rows)``, and ``params`` names the swept parameters
    carried on each row.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _rows_to_arrays(rows, obs, params)
    payload.update(
        {
            "schema_version": np.int64(SCHEMA_VERSION),
            "experiment": np.array(spec.name, dtype=_STR),
            "crn_group": np.array(spec.crn_group, dtype=_STR),
            "config_json": np.array(config_json(spec), dtype="U8192"),
            "code_fingerprint": np.array(code_fingerprint(), dtype=_STR),
            "param_names": np.array(list(params), dtype=_STR),
        }
    )
    for k, v in (diagnostics or {}).items():
        payload[k] = np.asarray(v)
    np.savez_compressed(path, **payload)
    return path


def config_json(spec):
    """The canonical description of a run, excluding execution parameters.

    Worker count and chunk size are deliberately absent: they must never enter
    the identity of a result.  The reference implementation hashed them into its
    cache key, so raising the worker count invalidated every cached society.
    """
    from dataclasses import asdict

    d = asdict(spec)
    return json.dumps(d, sort_keys=True, default=str)


def merge(shard_dir, out_path, expect_spec=None):
    """Combine every shard in a directory into one results file."""
    shard_dir, out_path = Path(shard_dir), Path(out_path)
    shards = sorted(shard_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")

    parts, meta = [], None
    for s in shards:
        with np.load(s, allow_pickle=False) as z:
            cur = {
                "config_json": str(z["config_json"]),
                "code_fingerprint": str(z["code_fingerprint"]),
                "obs_names": [str(x) for x in z["obs_names"]],
                "param_names": [str(x) for x in z["param_names"]],
            }
            if meta is None:
                meta = cur
                scalars = {
                    k: z[k]
                    for k in ("schema_version", "experiment", "crn_group")
                    if k in z.files
                }
            elif cur != meta:
                raise ValueError(
                    f"shard {s.name} disagrees with the others on configuration or "
                    "code fingerprint; refusing to merge incompatible runs"
                )
            parts.append({k: z[k] for k in z.files})

    keys = ["point_id", "replicate_dis", "replicate_init"] + meta["param_names"] + ["obs"]
    merged = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}

    # Deduplicate: a resumed run may re-simulate a chunk that had been written
    # but not recorded.  Identical keys must give identical results, so keeping
    # the first is safe.
    ident = np.array(
        [
            f"{a}|{b}|{c}"
            for a, b, c in zip(
                merged["point_id"], merged["replicate_dis"], merged["replicate_init"]
            )
        ]
    )
    _, first = np.unique(ident, return_index=True)
    order = np.sort(first)
    merged = {k: v[order] for k, v in merged.items()}

    merged["obs_names"] = np.array(meta["obs_names"], dtype=_STR)
    merged["param_names"] = np.array(meta["param_names"], dtype=_STR)
    merged["config_json"] = np.array(meta["config_json"], dtype="U8192")
    merged["code_fingerprint"] = np.array(meta["code_fingerprint"], dtype=_STR)
    merged.update(scalars)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **merged)
    return out_path


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _obs_column(res, name):
    names = [str(x) for x in res["obs_names"]]
    if name not in names:
        raise KeyError(f"{name!r} not among {names}")
    return res["obs"][:, names.index(name)]


def to_grid(res, name, axes=("f_d", "d")):
    """Pivot the long table into ``(n_rows, n_cols, n_replicates)``.

    Missing societies come back as NaN rather than silently shrinking the
    replicate count, so a partial run is visibly partial.
    """
    values = _obs_column(res, name)
    row_vals = np.unique(res[axes[0]])
    col_vals = np.unique(res[axes[1]])
    reps = np.unique(res["replicate_init"])

    ri = np.searchsorted(row_vals, res[axes[0]])
    ci = np.searchsorted(col_vals, res[axes[1]])
    pi = np.searchsorted(reps, res["replicate_init"])

    grid = np.full((row_vals.size, col_vals.size, reps.size), np.nan)
    grid[ri, ci, pi] = values
    return grid, row_vals, col_vals


def summarise(res, name, axes=("f_d", "d")):
    """Replicate mean and standard error for one observable."""
    grid, rows, cols = to_grid(res, name, axes)
    n = np.sum(~np.isnan(grid), axis=2)
    mean = np.nanmean(grid, axis=2)
    sd = np.nanstd(grid, axis=2, ddof=1)
    sem = sd / np.sqrt(np.maximum(n, 1))
    return {"mean": mean, "sem": sem, "sd": sd, "n": n, "rows": rows, "cols": cols}


def pending(all_keys, shard_dir):
    """Which of ``all_keys`` have not been simulated yet."""
    shard_dir = Path(shard_dir)
    done = set()
    for s in shard_dir.glob("*.npz"):
        try:
            with np.load(s, allow_pickle=False) as z:
                for pid, dis, ini in zip(
                    z["point_id"], z["replicate_dis"], z["replicate_init"]
                ):
                    done.add((str(pid), int(dis), int(ini)))
        except (OSError, KeyError, ValueError):
            # A shard truncated by a kill is simply redone.
            continue
    return [k for k in all_keys if (k.point_id, k.disorder, k.init) not in done]
