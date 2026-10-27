"""A small scientific smoke test for one periodic DPA4C checkpoint.

Only the first requested ``sog_candidates/train`` frames are read. A passing
result does not certify production use, grid convergence, learned long-range
gradients, physical electronic states or the upstream SOG implementation.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import json

import numpy as np

from cace_sog_training.data import IndexReader
from .predict import (_check_condition, _default_loader, _evaluate, _hash_file,
                      _index_identity, _model_details, _stat)


def _write_status(path, state):
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(state, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _validate_settings(condition, frames, atoms_per_frame, step, force_atol,
                       force_rtol, energy_atol, periodic_force_atol):
    condition = np.asarray(condition, dtype=np.float64)
    if condition.shape != (2,) or not np.isfinite(condition).all() or not np.array_equal(condition, [0., 1.]):
        raise ValueError("this workflow requires explicit --charge-spin 0 1")
    for name, value, limit in (("frames", frames, 256), ("atoms_per_frame", atoms_per_frame, 64)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer between 1 and {limit} for this smoke test")
    for name, value in (("step", step), ("force_atol", force_atol), ("energy_atol", energy_atol),
                        ("periodic_force_atol", periodic_force_atol)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(force_rtol) or force_rtol < 0:
        raise ValueError("force_rtol must be nonnegative and finite")
    return condition


def preflight(database, checkpoint, kind, output, charge_spin, *, frames=2, atoms_per_frame=2,
              step=0.001, force_atol=0.05, force_rtol=0.05, energy_atol=0.002,
              periodic_force_atol=0.002, loader=None):
    """Test energy-gradient forces and lattice translation with one model."""
    if kind not in ("dpa", "sog"):
        raise ValueError("kind must be dpa or sog")
    condition = _validate_settings(charge_spin, frames, atoms_per_frame, step, force_atol,
                                   force_rtol, energy_atol, periodic_force_atol)
    database = Path(database).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output in (database, checkpoint) or output.exists() or os.path.lexists(output):
        raise ValueError("report path exists or aliases an input; choose a NEW report")
    # Prevent writing the report into the original export, whose integrity the
    # IndexReader assumes while evaluation is running.
    with IndexReader(database) as reader:
        root = Path(reader.metadata["config"]["export_dir"]).resolve()
        if output == root or root in output.parents:
            raise ValueError("preflight report must be outside the source export directory")
        index_identity = _index_identity(reader.metadata)
        available = reader.counts("sog_candidates", "train")
        if not available:
            raise ValueError("the indexed sog_candidates/train split is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "kind": kind, "database": str(database),
             "index_identity": index_identity, "checkpoint": str(checkpoint),
             "channel": "sog_candidates", "split": "train", "test_split_read": False,
             "condition": condition.tolist(),
             "condition_provenance": "explicit model input assumption; not a measured DFT state",
             "settings": {"requested_frames": frames, "atoms_per_frame": atoms_per_frame,
                          "displacement_A": step, "force_atol_eV_A": force_atol,
                          "force_rtol": force_rtol, "periodic_energy_atol_eV": energy_atol,
                          "periodic_force_atol_eV_A": periodic_force_atol},
             "limitations": ["small numerical smoke test only, not a production certification",
                             "does not test grid convergence or nonzero long-range parameter gradients",
                             "does not prove electronic state or physical long-range significance"],
             "frames": [], "started_unix": time.time()}
    # Exclusive creation prevents two test processes from sharing a report.
    with output.open("x", encoding="utf-8") as marker:
        json.dump(state, marker, ensure_ascii=False, allow_nan=False)
        marker.flush()
        os.fsync(marker.fileno())
    model = None
    try:
        digest, checkpoint_stat = _hash_file(checkpoint)
        state["checkpoint_sha256"] = digest
        model = (loader or _default_loader)(checkpoint)
        details = _model_details(model, kind)
        state["model"] = details
        failures = []
        with IndexReader(database) as reader:
            if _index_identity(reader.metadata) != index_identity or reader.counts("sog_candidates", "train") != available:
                raise ValueError("training index identity changed before preflight")
            for ordinal, record in enumerate(reader.iter_records(["sog_candidates"], "train")):
                if ordinal >= frames:
                    break
                if record["channel"] != "sog_candidates" or record["split"] != "train" or record["ordinal"] != ordinal:
                    raise ValueError("training records are not in contiguous ordinal order")
                atoms = reader.read_atoms(record)
                _check_condition(atoms, condition)
                energy, force = _evaluate(model, details, atoms, condition, kind)
                force = np.asarray(force, dtype=np.float64)
                if force.shape != (len(atoms), 3) or not math.isfinite(energy) or not np.isfinite(force).all():
                    raise ValueError("nonfinite or invalid reference model output")
                atom_count = min(atoms_per_frame, len(atoms))
                differences = []
                for atom_index in range(atom_count):
                    for axis in range(3):
                        plus, minus = atoms.copy(), atoms.copy()
                        plus.positions[atom_index, axis] += step
                        minus.positions[atom_index, axis] -= step
                        e_plus, _ = _evaluate(model, details, plus, condition, kind)
                        e_minus, _ = _evaluate(model, details, minus, condition, kind)
                        numerical = -(e_plus - e_minus) / (2.0 * step)
                        predicted = float(force[atom_index, axis])
                        difference = abs(numerical - predicted)
                        threshold = force_atol + force_rtol * max(abs(numerical), abs(predicted))
                        ok = difference <= threshold
                        differences.append({"atom": atom_index, "axis": axis,
                                            "finite_difference_force_eV_A": numerical,
                                            "predicted_force_eV_A": predicted,
                                            "abs_difference_eV_A": difference,
                                            "allowed_abs_difference_eV_A": threshold, "passed": ok})
                        if not ok:
                            failures.append(f"frame {ordinal}, atom {atom_index}, axis {axis}: force finite difference")
                shifted = atoms.copy()
                shifted.positions[0] += np.asarray(atoms.cell, dtype=np.float64)[0]
                shifted_energy, shifted_force = _evaluate(model, details, shifted, condition, kind)
                shifted_force = np.asarray(shifted_force, dtype=np.float64)
                energy_difference = abs(shifted_energy - energy)
                force_difference = float(np.max(np.abs(shifted_force - force)))
                translation_ok = energy_difference <= energy_atol and force_difference <= periodic_force_atol
                if not translation_ok:
                    failures.append(f"frame {ordinal}: lattice translation invariance")
                state["frames"].append({"ordinal": ordinal, "record_id": int(record["id"]),
                    "frame_sha256": record["sha256"], "natoms": len(atoms),
                    "force_finite_differences": differences,
                    "translation": {"atom": 0, "cell_vector": 0,
                                    "energy_abs_difference_eV": energy_difference,
                                    "force_max_abs_difference_eV_A": force_difference,
                                    "passed": translation_ok}})
            reader.verify_sources()
        if len(state["frames"]) != min(frames, available):
            raise ValueError("training index contains fewer records than its metadata count")
        if _stat(checkpoint) != checkpoint_stat:
            raise ValueError("checkpoint changed during preflight")
        values = [v["abs_difference_eV_A"] for row in state["frames"] for v in row["force_finite_differences"]]
        state.update(frames_checked=len(state["frames"]), force_components_checked=len(values),
                     maximum_force_fd_abs_difference_eV_A=max(values),
                     maximum_translation_energy_abs_difference_eV=max(row["translation"]["energy_abs_difference_eV"] for row in state["frames"]),
                     maximum_translation_force_abs_difference_eV_A=max(row["translation"]["force_max_abs_difference_eV_A"] for row in state["frames"]),
                     failures=failures, finished_unix=time.time())
        if failures:
            raise ValueError(f"{len(failures)} scientific smoke checks exceeded their configured tolerances")
        state["status"] = "passed"
        _write_status(output, state)
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4096], finished_unix=time.time())
        try:
            _write_status(output, state)
        except OSError:
            pass
        raise
    finally:
        backend = getattr(model, "deep_eval", None)
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("dpa", "sog"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--charge-spin", required=True, nargs=2, type=float, metavar=("CHARGE", "MULTIPLICITY"))
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--atoms-per-frame", type=int, default=2)
    parser.add_argument("--step", type=float, default=0.001, help="central finite-difference displacement in angstrom")
    parser.add_argument("--force-atol", type=float, default=0.05, help="finite-difference absolute tolerance in eV/angstrom")
    parser.add_argument("--force-rtol", type=float, default=0.05)
    parser.add_argument("--energy-atol", type=float, default=0.002, help="periodic translation energy tolerance in eV per frame")
    parser.add_argument("--periodic-force-atol", type=float, default=0.002, help="periodic translation force tolerance in eV/angstrom")
    args = parser.parse_args(argv)
    try:
        result = preflight(args.database, args.model, args.kind, args.output, args.charge_spin,
            frames=args.frames, atoms_per_frame=args.atoms_per_frame, step=args.step,
            force_atol=args.force_atol, force_rtol=args.force_rtol, energy_atol=args.energy_atol,
            periodic_force_atol=args.periodic_force_atol)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; inspect the failed preflight report.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
