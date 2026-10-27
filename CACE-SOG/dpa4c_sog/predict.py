"""Paired, streaming DPA4C and DPA4C-SOG predictions on the original test split.

DeepPot 3.2 pt_expt accepts training checkpoints directly and forwards the
``charge_spin`` keyword through its public ``eval`` wrapper. The explicit
condition is a user-specified assumption, not a measured electronic state.
Predictions are written only as complete pairs and published without overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from cace_sog_training.data import IndexReader
from .convert import _check_condition as _validate_source_condition


CHANNEL, SPLIT = "sog_candidates", "test"


def _stat(path):
    info = path.stat()
    return (info.st_size, info.st_mtime_ns)


def _hash_file(path):
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if _stat(path) != before:
        raise ValueError(f"input changed while hashing: {path}")
    return digest.hexdigest(), before


def _index_identity(metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf8")).hexdigest()


def _model_details(model, label):
    """Fail visibly unless this backend can apply and identify the condition."""
    try:
        type_map = list(model.get_type_map())
        has_condition = model.has_chg_spin_ebd()
        config = model.get_model_def_script()
        inner_eval = model.deep_eval.eval
    except (AttributeError, NotImplementedError) as exc:
        raise ValueError(f"{label}: DeepPot backend cannot expose required model/charge_spin metadata") from exc
    if not has_condition:
        raise ValueError(f"{label}: checkpoint does not expose charge_spin conditioning")
    try:
        parameters = inspect.signature(inner_eval).parameters
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: cannot verify backend charge_spin support") from exc
    if "charge_spin" not in parameters:
        raise ValueError(f"{label}: backend eval does not explicitly support charge_spin")
    if not type_map or any(not isinstance(x, str) or not x for x in type_map) or len(set(type_map)) != len(type_map):
        raise ValueError(f"{label}: invalid or duplicate model type_map")
    if not isinstance(config, dict):
        raise ValueError(f"{label}: checkpoint model definition is unavailable")
    if isinstance(config.get("model"), dict):
        config = config["model"]
    descriptor = config.get("descriptor", {})
    if not isinstance(descriptor, dict) or str(descriptor.get("type", "")).lower() != "dpa4c":
        raise ValueError(f"{label}: expected DPA4C descriptor in checkpoint metadata")
    lr = config.get("long_range")
    if label == "dpa" and lr:
        raise ValueError("dpa checkpoint must be the short-range baseline, without a long-range module")
    if label == "sog":
        if not isinstance(lr, dict) or lr.get("type") != "latent_charge" or lr.get("periodic") is not True:
            raise ValueError("sog checkpoint must contain a periodic latent_charge long-range module")
        if not isinstance(lr.get("kernel"), dict) or lr["kernel"].get("type") != "sog":
            raise ValueError("sog checkpoint must contain a SOG kernel")
    for field in ("get_dim_fparam", "get_dim_aparam"):
        getter = getattr(model, field, None)
        if getter is None or int(getter()) != 0:
            raise ValueError(f"{label}: this predictor requires zero additional fparam/aparam dimensions")
    return {"type_map": type_map, "descriptor": "dpa4c", "has_charge_spin_embedding": True,
            "long_range": lr if lr else None, "additional_frame_and_atomic_parameters": 0}


def _evaluate(model, details, atoms, condition, label):
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    mapping = {symbol: index for index, symbol in enumerate(details["type_map"])}
    missing = set(symbols) - mapping.keys()
    if missing:
        raise ValueError(f"{label}: test frame contains elements missing from its type_map: {sorted(missing)}")
    types = np.asarray([mapping[symbol] for symbol in symbols], dtype=np.int32)
    if [details["type_map"][int(index)] for index in types] != symbols:
        raise ValueError(f"{label}: atom/type mapping does not preserve original atom order")
    coordinates = np.asarray(atoms.positions, dtype=np.float64).reshape(1, n, 3)
    cell = np.asarray(atoms.cell, dtype=np.float64).reshape(1, 9)
    if not np.isfinite(coordinates).all() or not np.isfinite(cell).all() or not np.asarray(atoms.pbc).all():
        raise ValueError("test frame requires finite coordinates and a fully periodic cell")
    try:
        output = model.eval(coords=coordinates.copy(), cells=cell.copy(), atom_types=types.copy(),
                            atomic=False, charge_spin=np.asarray(condition, dtype=np.float64).reshape(1, 2))
    except TypeError as exc:
        raise ValueError(f"{label}: DeepPot cannot evaluate an explicit charge_spin condition") from exc
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise ValueError(f"{label}: DeepPot did not return energy, force and virial")
    energy = np.asarray(output[0], dtype=np.float64)
    forces = np.asarray(output[1], dtype=np.float64)
    virial = np.asarray(output[2], dtype=np.float64)
    if energy.shape not in ((1,), (1, 1)) or forces.shape != (1, n, 3) or virial.shape not in ((1, 9), (1, 3, 3)):
        raise ValueError(f"{label}: unexpected DeepPot energy/force/virial shapes")
    if not (np.isfinite(energy).all() and np.isfinite(forces).all() and np.isfinite(virial).all()):
        raise ValueError(f"{label}: nonfinite model output")
    return float(energy.reshape(-1)[0]), forces[0].tolist()


def _check_condition(atoms, condition):
    # Explicit information in the source must never be silently overwritten.
    _validate_source_condition(atoms.info)
    for key in ("net_charge", "total_charge", "charge"):
        if key in atoms.info and not math.isclose(float(atoms.info[key]), condition[0], rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"frame {key} conflicts with requested constant charge")


def _status(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _default_loader(path):
    try:
        from deepmd.infer import DeepPot
    except ImportError as exc:
        raise RuntimeError("install the pinned DeePMD-kit pt_expt environment before prediction") from exc
    return DeepPot(str(path), auto_batch_size=False)


def predict(database, dpa, sog, output, charge_spin, *, loader=None, progress_seconds=30):
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
            raise ValueError("baseline and SOG checkpoints have identical SHA256")
        state["checkpoints"] = {"dpa": {"path": str(dpa), "sha256": dpa_sha},
                                "sog": {"path": str(sog), "sha256": sog_sha}}
        loader = loader or _default_loader
        base_model, sog_model = loader(dpa), loader(sog)
        models = {"dpa": _model_details(base_model, "dpa"), "sog": _model_details(sog_model, "sog")}
        if set(models["dpa"]["type_map"]) != set(models["sog"]["type_map"]):
            raise ValueError("baseline and SOG checkpoint type maps contain different elements")
        state["models"] = models
        with IndexReader(database) as reader:
            total = reader.counts(CHANNEL, SPLIT)
            if total == 0:
                raise ValueError("the indexed sog_candidates/test split is empty")
            state.update(index_identity=_index_identity(reader.metadata), index_path=str(database), expected_frames=total)
            handle, name = tempfile.mkstemp(prefix="." + output.name + ".partial-", dir=str(output.parent))
            partial = Path(name)
            state["partial_path"] = str(partial)
            last = time.monotonic()
            with os.fdopen(handle, "w", encoding="utf8", newline="\n") as stream:
                _status(status, state)
                for ordinal, record in enumerate(reader.iter_records([CHANNEL], SPLIT)):
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
                    base_e, base_f = _evaluate(base_model, models["dpa"], atoms, values, "dpa")
                    sog_e, sog_f = _evaluate(sog_model, models["sog"], atoms, values, "sog")
                    row = {"frame_key": f"{CHANNEL}/{SPLIT}/{ordinal}", "split": SPLIT,
                           "frame_sha256": record["sha256"], "natoms": len(atoms),
                           "reference_energy_eV": ref_e, "dpa_energy_eV": base_e, "sog_energy_eV": sog_e,
                           "reference_forces_eV_A": ref_f.tolist(), "dpa_forces_eV_A": base_f,
                           "sog_forces_eV_A": sog_f}
                    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    state["frames_written"] += 1
                    if time.monotonic() - last >= progress_seconds:
                        print(f"Predicted {state['frames_written']:,} / {total:,} test frames", file=sys.stderr, flush=True)
                        last = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
            if state["frames_written"] != total:
                raise ValueError("indexed count and number of predicted test frames differ")
            reader.verify_sources()
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
    parser.add_argument("--dpa", required=True, type=Path)
    parser.add_argument("--sog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--charge-spin", nargs=2, type=float, required=True, metavar=("CHARGE", "MULTIPLICITY"))
    parser.add_argument("--progress-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        result = predict(args.database, args.dpa, args.sog, args.output, args.charge_spin,
                         progress_seconds=args.progress_seconds)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Prediction interrupted; inspect the failed status and partial file before using a new output path.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
