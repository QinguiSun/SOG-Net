"""Paired, bounded-memory energy/force evaluation for one untouched test split.

The input JSONL must contain one *paired* record per frame, with keys:
frame_key, natoms, reference_energy_eV, dpa_energy_eV, sog_energy_eV,
reference_forces_eV_A, dpa_forces_eV_A, sog_forces_eV_A. The optional split
must equal test. Optional frame_sha256, source_frame_sha256, or sha256 fields
are checked for syntax and agreement only; the producer must verify the original
source frame independently.

All frames and force components enter the metrics and the 2D density plots.
No input arrays are retained between frames. A temporary SQLite primary key
detects duplicate frame identifiers without growing RAM with dataset size.
If <input>.status.json exists it must be a completed prediction manifest with
matching full-file SHA256 and frame counts. Standalone JSONL needs no sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np

MAX_LINE_BYTES = 64 * 1024 * 1024
MAX_SIDECAR_BYTES = 4 * 1024 * 1024
MAX_ATOMS = 100000
MODEL_NAMES = ("dpa", "sog")


class RunningStats:
    """Reference central moment and residual moments with O(1) state."""

    def __init__(self):
        self.n = 0
        self.reference_mean = 0.0
        self.reference_m2 = 0.0
        self.residual_sum = 0.0
        self.absolute_sum = 0.0
        self.squared_sum = 0.0

    def add(self, reference, prediction):
        reference = np.asarray(reference, dtype=np.float64).reshape(-1)
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        if reference.size == 0 or reference.shape != prediction.shape:
            raise ValueError("prediction/reference arrays must have the same nonzero size")
        if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
            raise ValueError("nonfinite reference/prediction")
        delta = prediction - reference
        batch_n = int(reference.size)
        batch_mean = float(np.mean(reference))
        batch_m2 = float(np.sum((reference - batch_mean) ** 2))
        total = self.n + batch_n
        shift = batch_mean - self.reference_mean
        self.reference_m2 += batch_m2 + shift * shift * self.n * batch_n / total
        self.reference_mean += shift * batch_n / total
        self.residual_sum += float(np.sum(delta))
        self.absolute_sum += float(np.sum(np.abs(delta)))
        self.squared_sum += float(np.dot(delta, delta))
        self.n = total
        if not all(math.isfinite(x) for x in (self.reference_mean, self.reference_m2,
                self.residual_sum, self.absolute_sum, self.squared_sum)):
            raise ValueError("numeric overflow while accumulating statistics")

    def result(self, unit):
        if not self.n:
            raise ValueError("no evaluation values")
        return {"count": self.n, "unit": unit,
                "mean_difference": self.residual_sum / self.n,
                "mae": self.absolute_sum / self.n,
                "rmse": math.sqrt(self.squared_sum / self.n),
                "r2": 1.0 - self.squared_sum / self.reference_m2 if self.reference_m2 > 0 else None}


def _scalar(record, key):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid finite number for " + key)
    return float(value)


def _force(record, key, natoms):
    value = record.get(key)
    if not isinstance(value, list) or len(value) != natoms:
        raise ValueError(key + " must have shape [natoms, 3]")
    if any(not isinstance(row, list) or len(row) != 3 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) for x in row) for row in value):
        raise ValueError(key + " must contain numeric three-component vectors")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (natoms, 3) or not np.isfinite(result).all():
        raise ValueError(key + " contains nonfinite values or wrong shape")
    return result.reshape(-1)


def _record(raw, lineno):
    try:
        item = json.loads(raw.decode("utf-8"))
        if not isinstance(item, dict):
            raise ValueError("record must be a JSON object")
        key = item.get("frame_key")
        if not isinstance(key, str) or not key or len(key) > 4096:
            raise ValueError("frame_key must be a nonempty string of at most 4096 characters")
        if "split" in item and item["split"] != "test":
            raise ValueError("only the held-out test split may be evaluated here")
        digests = []
        for name in ("frame_sha256", "source_frame_sha256", "sha256"):
            if name not in item:
                continue
            digest = item[name]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(name + " must be 64 lowercase hexadecimal characters")
            digests.append(digest)
        if len(set(digests)) > 1:
            raise ValueError("source frame SHA256 fields disagree")
        natoms = item.get("natoms")
        if isinstance(natoms, bool) or not isinstance(natoms, int) or not 1 <= natoms <= MAX_ATOMS:
            raise ValueError("natoms must be an integer in the supported range")
        energy = {"ref": _scalar(item, "reference_energy_eV") / natoms}
        forces = {"ref": _force(item, "reference_forces_eV_A", natoms)}
        for model in MODEL_NAMES:
            energy[model] = _scalar(item, model + "_energy_eV") / natoms
            forces[model] = _force(item, model + "_forces_eV_A", natoms)
        return key, natoms, energy, forces
    except (UnicodeError, ValueError, TypeError, OverflowError) as exc:
        raise ValueError("input line %d: %s" % (lineno, exc)) from exc


def _lines(path, digest):
    with path.open("rb", buffering=1024 * 1024) as stream:
        lineno = 0
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            lineno += 1
            if len(raw) > MAX_LINE_BYTES:
                raise ValueError("input line %d exceeds the frame-size limit" % lineno)
            digest.update(raw)
            if not raw.strip():
                raise ValueError("blank input line %d" % lineno)
            yield _record(raw, lineno)


def _stat(path):
    value = path.stat()
    return value.st_size, value.st_mtime_ns


def _sidecar(path):
    """Read an optional bounded prediction status, preserving its provenance."""
    status_path = path.with_name(path.name + ".status.json")
    try:
        before = _stat(status_path)
    except FileNotFoundError:
        return None
    if before[0] > MAX_SIDECAR_BYTES:
        raise ValueError("prediction status sidecar exceeds the size limit")
    with status_path.open("rb") as stream:
        raw = stream.read(MAX_SIDECAR_BYTES + 1)
    if len(raw) > MAX_SIDECAR_BYTES or _stat(status_path) != before:
        raise ValueError("prediction status sidecar changed or exceeds the size limit")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid prediction status sidecar") from exc
    if not isinstance(data, dict) or data.get("status") != "complete":
        raise ValueError("prediction status sidecar must report complete")
    if data.get("split", "test") != "test" or data.get("channel", "sog_candidates") != "sog_candidates":
        raise ValueError("prediction status sidecar identifies a different channel or split")
    expected, written = data.get("expected_frames"), data.get("frames_written")
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (expected, written)) or expected != written:
        raise ValueError("prediction status sidecar frame counts are missing, invalid or incomplete")
    digest = data.get("predictions_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("prediction status sidecar has an invalid predictions_sha256")
    return {"path": str(status_path), "stat": before, "sha256": hashlib.sha256(raw).hexdigest(), "content": data}


def _check_sidecar_unchanged(source, initial):
    current = _sidecar(source)
    if initial is None and current is None:
        return
    if initial is None or current is None or current["stat"] != initial["stat"] or current["sha256"] != initial["sha256"]:
        raise ValueError("prediction status sidecar changed during evaluation")


def _span(bounds):
    lo, hi = bounds
    if lo == hi:
        margin = max(abs(lo) * 1e-6, 1e-8)
        return lo - margin, hi + margin
    margin = (hi - lo) * 0.005
    return lo - margin, hi + margin


def _accumulate(histogram, reference, predicted, lo, hi):
    bins = histogram.shape[0]
    scale = bins / (hi - lo)
    # Values on the upper edge belong to the final bin, not an overflow bin.
    x = np.clip(((np.asarray(reference) - lo) * scale).astype(np.int64), 0, bins - 1)
    y = np.clip(((np.asarray(predicted) - lo) * scale).astype(np.int64), 0, bins - 1)
    np.add.at(histogram, (x, y), 1)


def _render(path, model, frame_count, bounds, histograms, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    title = {"dpa": "DPA4c", "sog": "DPA4c-SOG"}[model]
    fig.suptitle("%s: held-out test | %s structures" % (title, format(frame_count, ",")))
    for ax, prop, unit in ((axes[0], "energy", "eV/atom"), (axes[1], "force", "eV/Å")):
        lo, hi = bounds[prop]
        grid = histograms[model][prop]
        masked = np.ma.masked_equal(grid.T, 0)
        image = ax.imshow(masked, origin="lower", extent=(lo, hi, lo, hi),
                          interpolation="nearest", aspect="equal", cmap="viridis",
                          norm=LogNorm(vmin=1, vmax=max(2, int(grid.max()))))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)
        ax.set_xlabel("DFT %s (%s)" % ("energy per atom" if prop == "energy" else "force component", unit))
        ax.set_ylabel("%s predicted %s (%s)" % (title, "energy per atom" if prop == "energy" else "force component", unit))
        ax.set_title("Energy comparison" if prop == "energy" else "Force comparison")
        row = metrics[model][prop]
        r2 = "undefined" if row["r2"] is None else "%.5g" % row["r2"]
        annotation = "Mean difference = %.5g\nMAE = %.5g\nRMSE = %.5g\nR² = %s" % (
            row["mean_difference"], row["mae"], row["rmse"], r2)
        ax.text(0.02, 0.98, annotation, transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        fig.colorbar(image, ax=ax, label="Values per bin (log scale)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_predictions(input_path, output_dir, bins=200):
    """Evaluate exactly paired predictions in two complete streaming passes."""
    if isinstance(bins, bool) or not isinstance(bins, int) or not 16 <= bins <= 1024:
        raise ValueError("bins must be an integer between 16 and 1024")
    source = Path(input_path).expanduser().resolve(strict=True)
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise ValueError("output directory exists; choose a new directory")
    if not source.is_file():
        raise ValueError("input must be a regular JSONL file")
    target.parent.mkdir(parents=True, exist_ok=True)
    initial_stat = _stat(source)
    sidecar = _sidecar(source)
    with tempfile.TemporaryDirectory(prefix=".dpa4c-eval-", dir=str(target.parent)) as temp:
        temp = Path(temp)
        stage = temp / "result"
        stage.mkdir()
        db = sqlite3.connect(str(temp / "seen.sqlite3"))
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA mmap_size=0")
        db.execute("CREATE TABLE seen(frame_key TEXT PRIMARY KEY) WITHOUT ROWID")
        stats = {model: {prop: RunningStats() for prop in ("energy", "force")} for model in MODEL_NAMES}
        bounds = {prop: [math.inf, -math.inf] for prop in ("energy", "force")}
        count, atom_count, last = 0, 0, time.monotonic()
        digest1 = hashlib.sha256()
        try:
            for key, natoms, energy, forces in _lines(source, digest1):
                try:
                    db.execute("INSERT INTO seen VALUES (?)", (key,))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("duplicate frame_key: " + key) from exc
                count += 1
                atom_count += natoms
                for prop, values in (("energy", energy), ("force", forces)):
                    for val in values.values():
                        bounds[prop][0] = min(bounds[prop][0], float(np.min(val)))
                        bounds[prop][1] = max(bounds[prop][1], float(np.max(val)))
                    for model in MODEL_NAMES:
                        stats[model][prop].add([values["ref"]] if prop == "energy" else values["ref"],
                                               [values[model]] if prop == "energy" else values[model])
                if count % 10000 == 0:
                    db.commit()
                if time.monotonic() - last >= 30:
                    print("metrics: %s paired test frames" % format(count, ","), file=sys.stderr, flush=True)
                    last = time.monotonic()
            db.commit()
        finally:
            db.close()
        if not count:
            raise ValueError("input contains no paired test frames")
        if _stat(source) != initial_stat:
            raise ValueError("input changed during metrics pass")
        if sidecar is not None and (sidecar["content"]["frames_written"] != count or
                                    sidecar["content"]["expected_frames"] != count or
                                    sidecar["content"]["predictions_sha256"] != digest1.hexdigest()):
            raise ValueError("prediction status sidecar does not match input frame count or SHA256")
        _check_sidecar_unchanged(source, sidecar)
        bounds = {name: _span(value) for name, value in bounds.items()}
        histograms = {model: {prop: np.zeros((bins, bins), dtype=np.uint64)
                              for prop in ("energy", "force")} for model in MODEL_NAMES}
        digest2 = hashlib.sha256()
        count2, last = 0, time.monotonic()
        for _, _, energy, forces in _lines(source, digest2):
            count2 += 1
            for prop, values in (("energy", energy), ("force", forces)):
                for model in MODEL_NAMES:
                    _accumulate(histograms[model][prop], values["ref"], values[model], *bounds[prop])
            if time.monotonic() - last >= 30:
                print("plot density: %s paired test frames" % format(count2, ","), file=sys.stderr, flush=True)
                last = time.monotonic()
        if count2 != count or digest1.hexdigest() != digest2.hexdigest() or _stat(source) != initial_stat:
            raise ValueError("input changed between metrics and plot passes")
        metrics = {model: {"energy": stats[model]["energy"].result("eV/atom"),
                           "force": stats[model]["force"].result("eV/Å")}
                   for model in MODEL_NAMES}
        for model in MODEL_NAMES:
            _render(stage / (model + "_parity.png"), model, count, bounds, histograms, metrics)
        if _stat(source) != initial_stat:
            raise ValueError("input changed while plots were being written")
        _check_sidecar_unchanged(source, sidecar)
        report = {"status": "complete", "split": "test", "paired_frames": count,
                  "atoms": atom_count, "force_components": 3 * atom_count,
                  "difference_definition": "prediction minus DFT reference",
                  "energy_definition": "total energy divided by number of atoms, one value per frame",
                  "force_definition": "all Cartesian force components, one value per component",
                  "r2_definition": "1 - sum((prediction-reference)^2) / sum((reference-mean(reference))^2); null when reference variance is zero",
                  "input_path": str(source), "input_size_bytes": initial_stat[0],
                  "input_sha256": digest1.hexdigest(), "metrics": metrics,
                  "prediction_provenance": None if sidecar is None else {
                      "status_path": sidecar["path"], "status_sha256": sidecar["sha256"],
                      "prediction_sha256_and_frame_counts_verified": True, "status": sidecar["content"]},
                  "plot": {"kind": "full-data 2D histogram", "bins_per_axis": bins,
                           "energy_range_eV_atom": list(bounds["energy"]), "force_range_eV_A": list(bounds["force"]),
                           "files": {"dpa": "dpa_parity.png", "sog": "sog_parity.png"}},
                  "provenance_note": "frame_key uniqueness and optional test split/hash syntax were checked; source frame and split identity must be verified by the prediction producer"}
        (stage / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if target.exists():
            raise ValueError("output directory appeared during evaluation")
        os.rename(str(stage), str(target))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare paired DPA4c and DPA4c-SOG predictions on the held-out test split.")
    parser.add_argument("--input", required=True, type=Path, help="one paired JSON object per frame")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        report = evaluate_predictions(args.input, args.output, args.bins)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
