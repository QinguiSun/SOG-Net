"""Evaluate raw pretrained and validation-selected fine-tuned DPA4C on held-out test."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from cace_sog_training.data import IndexReader
from .predict import (CHANNEL, SPLIT, _stat, _hash_file, _index_identity, _model_details,
                      _compare_architecture, _evaluate, _check_condition, _status, _default_loader,
                      _validation_frames, _verified_validation)

def predict(database, dpa, sog, output, charge_spin, *, validation_report=None, loader=None, progress_seconds=30, start=0, end=None):
    """Evaluate all original test frames; model loader is injectable for tests."""
    database, dpa, sog = (Path(x).expanduser().resolve(strict=True) for x in (database, dpa, sog))
    output = Path(output).expanduser().resolve()
    if any(output == p for p in (database, dpa, sog)):
        raise ValueError("output cannot overwrite an input")
    values = np.asarray(charge_spin, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or not np.array_equal(values, [0., 1.]):
        raise ValueError("this training conversion requires the explicit --charge-spin 0 1 assumption")
    if progress_seconds <= 0 or not math.isfinite(progress_seconds):
        raise ValueError("progress_seconds must be finite and positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    status = output.with_name(output.name + ".status.json")
    if output.exists():
        raise ValueError("prediction output already exists; choose a new path")
    # Exclusive status creation prevents concurrent writers sharing a target.
    with status.open("x", encoding="utf8") as marker:
        json.dump({"status": "running", "output": str(output)}, marker)
        marker.flush()
        os.fsync(marker.fileno())
    partial = None
    state = {"status": "running", "channel": CHANNEL, "split": SPLIT, "output": str(output),
             "charge_spin": values.tolist(),
             "condition_provenance": "explicit constant user assumption; not a measured electronic state",
             "frames_written": 0, "started_unix": time.time()}
    try:
        dpa_sha, dpa_stat = _hash_file(dpa)
        sog_sha, sog_stat = _hash_file(sog)
        if dpa_sha == sog_sha:
            raise ValueError("pretrained and fine-tuned checkpoints have identical SHA256")
        if validation_report is None:
            raise ValueError("full fine-tuned validation report is required before test")
        state["checkpoints"] = {"pretrained": {"path": str(dpa), "sha256": dpa_sha},
                                "finetuned": {"path": str(sog), "sha256": sog_sha}}
        loader = loader or _default_loader
        base_model, sog_model = loader(dpa), loader(sog)
        models = {"dpa": _model_details(base_model, "dpa"), "sog": _model_details(sog_model, "dpa")}
        if set(models["dpa"]["type_map"]) != set(models["sog"]["type_map"]):
            raise ValueError("checkpoint type maps contain different elements")
        state["architecture_comparison"] = _compare_architecture(models)
        state["models"] = {"pretrained": models["dpa"], "finetuned": models["sog"]}
        with IndexReader(database) as reader:
            total = reader.counts(CHANNEL, SPLIT)
            if total == 0:
                raise ValueError("the indexed sog_candidates/test split is empty")
            if end is None:
                end = total
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end)) or not 0 <= start < end <= total:
                raise ValueError("invalid test ordinal interval")
            identity = _index_identity(reader.metadata)
            index_stat = _stat(reader.database)
            val_count, val_digest = _validation_frames(reader)
            validation = {}
            report_stats = {}
            validation["finetuned"], report_stats["finetuned"] = _verified_validation(
                validation_report, "dpa", sog_sha, identity, val_count, val_digest)
            state.update(index_identity=identity, index_path=str(database), expected_frames=end-start,
                         test_total_frames=total, ordinal_start=start, ordinal_end=end,
                         is_shard=(start != 0 or end != total),
                         validation_provenance=validation, fine_tune_parent_not_verified=True,
                         fine_tune_note="fine-tuning provenance requires separate verification; baseline is the raw pretrained checkpoint")
            handle, name = tempfile.mkstemp(prefix="." + output.name + ".partial-", dir=str(output.parent))
            partial = Path(name)
            state["partial_path"] = str(partial)
            last = time.monotonic()
            with os.fdopen(handle, "w", encoding="utf8", newline="\n") as stream:
                _status(status, state)
                for ordinal in range(start, end):
                    record = reader.record(CHANNEL, SPLIT, ordinal)
                    if record["channel"] != CHANNEL or record["split"] != SPLIT or record["ordinal"] != ordinal:
                        raise ValueError("index test records are not in contiguous ordinal order")
                    atoms = reader.read_atoms(record)  # includes raw-frame SHA256 verification
                    if len(atoms) != record["natoms"]:
                        raise ValueError("index and source frame atom counts differ")
                    _check_condition(atoms, values)
                    ref_e = float(atoms.info["REF_energy"])
                    ref_f = np.asarray(atoms.arrays["REF_forces"], dtype=np.float64)
                    if not math.isfinite(ref_e) or ref_f.shape != (len(atoms), 3) or not np.isfinite(ref_f).all():
                        raise ValueError("invalid or nonfinite reference energy/forces")
                    base_e, base_f = _evaluate(base_model, models["dpa"], atoms, values, "pretrained")
                    sog_e, sog_f = _evaluate(sog_model, models["sog"], atoms, values, "finetuned")
                    row = {"frame_key": f"{CHANNEL}/{SPLIT}/{ordinal}", "split": SPLIT,
                           "frame_sha256": record["sha256"], "natoms": len(atoms),
                           "reference_energy_eV": ref_e, "pretrained_energy_eV": base_e, "finetuned_energy_eV": sog_e,
                           "reference_forces_eV_A": ref_f.tolist(), "pretrained_forces_eV_A": base_f,
                           "finetuned_forces_eV_A": sog_f}
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    state["frames_written"] += 1
                    if time.monotonic() - last >= progress_seconds:
                        print(f"Predicted {state['frames_written']:,} / {end-start:,} test frames in shard", file=sys.stderr, flush=True)
                        last = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if state["frames_written"] != end-start:
                raise ValueError("indexed count and number of predicted test frames differ")
            reader.verify_sources()
            if _stat(reader.database) != index_stat:
                raise ValueError("bound index changed during prediction")
        for label, info in state["validation_provenance"].items():
            if _stat(Path(info["path"])) != report_stats[label]:
                raise ValueError(f"{label}: validation report changed during prediction")
        if _stat(dpa) != dpa_stat or _stat(sog) != sog_stat:
            raise ValueError("checkpoint changed during evaluation")
        result_sha, _ = _hash_file(partial)
        # Atomic no-overwrite publication, followed by the completion marker.
        os.link(partial, output)
        partial.unlink()
        state.update(status="complete", predictions_sha256=result_sha, finished_unix=time.time())
        state.pop("partial_path", None)
        _status(status, state)
        return state
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:4096], finished_unix=time.time())
        try:
            _status(status, state)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--finetuned", required=True, type=Path)
    parser.add_argument("--finetuned-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--charge-spin", nargs=2, type=float, required=True)
    parser.add_argument("--progress-seconds", type=float, default=30)
    parser.add_argument("--start", type=int, default=0, help="inclusive test ordinal")
    parser.add_argument("--end", type=int, help="exclusive test ordinal")
    args = parser.parse_args(argv)
    try:
        result = predict(args.database, args.pretrained, args.finetuned, args.output, args.charge_spin,
                         validation_report=args.finetuned_validation, progress_seconds=args.progress_seconds,
                         start=args.start, end=args.end)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
