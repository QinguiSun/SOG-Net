"""Train-only elemental energy offsets, usable on Windows without PyTorch."""
import json
import math
import time

import numpy as np


def fit_baseline(reader, channels, progress_seconds=30):
    """Fit from indexed train labels with fixed-size normal sums, never ASE I/O."""
    gram = np.zeros((119, 119), dtype=np.float64)
    rhs = np.zeros(119, dtype=np.float64)
    elements = set()
    frames, last = 0, time.monotonic()
    for record in reader.iter_records(channels, "train"):
        counts = json.loads(record["counts_json"])
        zs = np.array([int(z) for z in counts], dtype=int)
        if not len(zs) or np.any(zs < 1) or np.any(zs > 118):
            raise ValueError("invalid atomic numbers in index")
        values = np.array([counts[str(z)] for z in zs], dtype=np.float64) / record["natoms"]
        gram[np.ix_(zs, zs)] += np.outer(values, values)
        rhs[zs] += values * (record["energy"] / record["natoms"])
        elements.update(int(z) for z in zs)
        frames += 1
        if time.monotonic() - last >= progress_seconds:
            print(f"train-only energy baseline: {frames:,} frames", flush=True)
            last = time.monotonic()
    if not frames:
        raise ValueError("selected train is empty")
    zs = sorted(elements)
    coefficients, _, rank, singular = np.linalg.lstsq(gram[np.ix_(zs, zs)], rhs[zs], rcond=1e-12)
    e0 = {z: float(value) for z, value in zip(zs, coefficients)}
    return e0, {"frames": frames, "elements": zs, "normal_matrix_rank": int(rank),
                "normal_matrix_singular_values": singular.tolist(),
                "method": "train-only per-atom weighted least squares; minimum-norm solution if rank deficient"}


def prepared_baseline(reader, channels):
    """Return a sealed fit only for its exact channel scope; otherwise refit."""
    prepared = reader.metadata.get("prepared_baseline")
    if not prepared or set(prepared.get("channels", ())) != set(channels):
        return None
    e0, fit = prepared["e0"], prepared["fit"]
    expected = reader.elements(channels, "train")
    if (set(map(int, e0)) != set(expected) or
            fit["frames"] != sum(reader.counts(c, "train") for c in channels) or
            not all(math.isfinite(float(v)) for v in e0.values())):
        raise ValueError("invalid prepared train-only energy baseline")
    return {int(z): float(v) for z, v in e0.items()}, dict(fit)
