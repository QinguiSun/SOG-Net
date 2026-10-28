"""Stream all original sog_candidates/val frames and select by force MAE.

One checkpoint is loaded at a time. This command never reads the test split.
The constant [0, 1] model condition is an assumption, not a DFT measurement.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from cace_sog_training.data import IndexReader
from .metrics import RunningStats
from .predict import (_check_condition, _default_loader, _evaluate, _hash_file,
                      _index_identity, _model_details, _stat, _status)


def validate(database, kind, checkpoints, output, charge_spin, *, loader=None, progress_seconds=30):
    """Evaluate all candidate checkpoints on identical original val records."""
    if kind not in ("dpa", "sog"):
        raise ValueError("kind must be dpa or sog")
    condition = np.asarray(charge_spin, dtype=np.float64)
    if condition.shape != (2,) or not np.isfinite(condition).all() or not np.array_equal(condition, [0., 1.]):
        raise ValueError("this conversion requires explicit --charge-spin 0 1")
    if not checkpoints:
        raise ValueError("provide at least one --checkpoint")
    if not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("progress_seconds must be finite and positive")
    database = Path(database).expanduser().resolve(strict=True)
    candidates = [Path(p).expanduser().resolve(strict=True) for p in checkpoints]
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError("validation output directory exists; choose a new directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()  # exclusive; concurrent runs cannot share an output directory
    state_path = target / "validation_status.json"
    state = {"status": "partial", "kind": kind, "channel": "sog_candidates", "split": "val",
             "test_split_read": False, "condition": condition.tolist(),
             "condition_provenance": "explicit model input assumption; not a measured electronic state",
             "database": str(database), "candidates_completed": 0, "started_unix": time.time()}
    _status(state_path, state)
    loader = loader or _default_loader
    model = None
    try:
        results = []
        expected, identity, frame_identity, first_type_set = None, None, None, None
        for candidate_index, checkpoint in enumerate(candidates):
            digest, before = _hash_file(checkpoint)
            model = loader(checkpoint)
            details = _model_details(model, kind)
            if first_type_set is None:
                first_type_set = set(details["type_map"])
            elif first_type_set != set(details["type_map"]):
                raise ValueError("checkpoint type maps contain different element sets")
            energy, force = RunningStats(), RunningStats()
            with IndexReader(database) as reader:
                n_frames = reader.counts("sog_candidates", "val")
                current_identity = _index_identity(reader.metadata)
                if n_frames <= 0:
                    raise ValueError("the indexed sog_candidates/val split is empty")
                if expected is None:
                    expected, identity = n_frames, current_identity
                elif expected != n_frames or identity != current_identity:
                    raise ValueError("validation index metadata or frame counts changed between checkpoints")
                count, last = 0, time.monotonic()
                frame_digest = hashlib.sha256()
                for ordinal, record in enumerate(reader.iter_records(["sog_candidates"], "val")):
                    if record["channel"] != "sog_candidates" or record["split"] != "val" or record["ordinal"] != ordinal:
                        raise ValueError("validation records are not in contiguous ordinal order")
                    frame_digest.update(json.dumps([ordinal, record["id"], record["sha256"],
                        record["natoms"], record["component_id"]], separators=(",", ":"),
                        ensure_ascii=True).encode("ascii") + b"\n")
                    atoms = reader.read_atoms(record)  # verifies source bytes against index SHA256
                    _check_condition(atoms, condition)
                    n = len(atoms)
                    if n != record["natoms"] or n <= 0:
                        raise ValueError("validation frame atom count mismatch")
                    reference_energy = float(atoms.info["REF_energy"])
                    reference_force = np.asarray(atoms.arrays["REF_forces"], dtype=np.float64)
                    if not math.isfinite(reference_energy) or reference_force.shape != (n, 3) or not np.isfinite(reference_force).all():
                        raise ValueError("invalid validation reference energy or forces")
                    predicted_energy, predicted_force = _evaluate(model, details, atoms, condition, kind)
                    energy.add([reference_energy / n], [predicted_energy / n])
                    force.add(reference_force, predicted_force)
                    count += 1
                    if time.monotonic() - last >= progress_seconds:
                        print(f"Validation checkpoint {candidate_index + 1}/{len(candidates)}: {count:,}/{n_frames:,} frames",
                              file=sys.stderr, flush=True)
                        last = time.monotonic()
                if count != n_frames:
                    raise ValueError("validation frame count differs from complete index")
                if frame_identity is None:
                    frame_identity = frame_digest.hexdigest()
                elif frame_identity != frame_digest.hexdigest():
                    raise ValueError("validation frame identities changed between checkpoints")
                reader.verify_sources()
            if _stat(checkpoint) != before:
                raise ValueError("checkpoint changed during validation")
            results.append({"checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                            "model": details, "frames": count, "frame_identity_sha256": frame_identity,
                            "energy": energy.result("eV/atom"), "force": force.result("eV/Å")})
            state.update(candidates_completed=candidate_index + 1, index_identity=identity, expected_frames=expected)
            _status(state_path, state)
            backend = getattr(model, "deep_eval", None)
            close = getattr(backend, "close", None)
            if callable(close):
                close()
            del close, backend
            model = None
            gc.collect()
        # Python min keeps the first candidate on an exact numerical tie.
        best = min(range(len(results)), key=lambda i: results[i]["force"]["mae"])
        report = {"status": "complete", "kind": kind, "channel": "sog_candidates", "split": "val",
                  "test_split_read": False, "index_path": str(database), "index_identity": identity,
                  "frames_per_checkpoint": expected, "frame_identity_sha256": frame_identity, "condition": condition.tolist(),
                  "condition_provenance": state["condition_provenance"],
                  "difference_definition": "prediction minus DFT reference",
                  "selection": "minimum full-validation force-component MAE; ties use input order",
                  "selected_index": best, "selected_checkpoint": results[best]["checkpoint"],
                  "selected_checkpoint_sha256": results[best]["checkpoint_sha256"],
                  "candidates": results}
        _status(target / "validation.json", report)
        state.update(status="complete", selected_checkpoint=report["selected_checkpoint"],
                     selected_checkpoint_sha256=report["selected_checkpoint_sha256"], finished_unix=time.time())
        _status(state_path, state)
        return report
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4096], finished_unix=time.time())
        try:
            _status(state_path, state)
        except OSError:
            pass
        raise
    finally:
        if model is not None:
            backend = getattr(model, "deep_eval", None)
            close = getattr(backend, "close", None)
            if callable(close):
                close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("dpa", "sog"))
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--charge-spin", nargs=2, required=True, type=float, metavar=("CHARGE", "MULTIPLICITY"))
    parser.add_argument("--progress-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        report = validate(args.database, args.kind, args.checkpoint, args.output, args.charge_spin,
                          progress_seconds=args.progress_seconds)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Validation interrupted; its result is incomplete. Choose a new output directory to retry.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
