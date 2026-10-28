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
    fitting = config.get("fitting_net")
    if fitting is not None and not isinstance(fitting, dict):
        raise ValueError(f"{label}: invalid fitting_net checkpoint metadata")
    return {"type_map": type_map, "descriptor": "dpa4c", "has_charge_spin_embedding": True,
            "descriptor_metadata": descriptor, "fitting_net_metadata": fitting,
            "long_range": lr if lr else None, "additional_frame_and_atomic_parameters": 0}


def _compare_architecture(models):
    """Compare shared, reported model fields without assuming default expansion."""
    checked, missing = [], []
    for section, field in (("descriptor", "channels"), ("descriptor", "lmax"),
                           ("descriptor", "radial_modes"), ("descriptor", "rcut"),
                           ("fitting_net", "neuron")):
        name = section + "." + field
        key = section + "_metadata"
        left = models["dpa"].get(key) or {}
        right = models["sog"].get(key) or {}
        if field not in left or field not in right:
            missing.append(name)
            continue
        left_value, right_value = left[field], right[field]
        if field == "neuron":
            if not isinstance(left_value, list) or not isinstance(right_value, list):
                raise ValueError("fitting_net.neuron metadata must be a list on both models")
        elif (isinstance(left_value, bool) or isinstance(right_value, bool) or
              not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)) or
              not math.isfinite(left_value) or not math.isfinite(right_value)):
            raise ValueError(f"{name} metadata must be finite numbers on both models")
        if left_value != right_value:
            raise ValueError(f"baseline and SOG architecture mismatch: {name}")
        checked.append(name)
    return {"status": "partially_compared" if missing else "matched_for_checked_fields",
            "matched_fields": checked, "not_compared_fields": missing,
            "note": "Only these fields were compared when reported by both checkpoints; this does not verify initialization or full architecture identity."}


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


def _validation_frames(reader):
    """Recreate the identity used by the streaming validation report."""
    expected = reader.counts(CHANNEL, "val")
    if expected <= 0:
        raise ValueError("the indexed sog_candidates/val split is empty")
    digest = hashlib.sha256()
    count = 0
    for ordinal, record in enumerate(reader.iter_records([CHANNEL], "val")):
        if record["channel"] != CHANNEL or record["split"] != "val" or record["ordinal"] != ordinal:
            raise ValueError("validation records are not in contiguous ordinal order")
        digest.update(json.dumps([ordinal, record["id"], record["sha256"],
            record["natoms"], record["component_id"]], separators=(",", ":"),
            ensure_ascii=True).encode("ascii") + b"\n")
        count += 1
    if count != expected:
        raise ValueError("validation record count differs from the complete index")
    reader.verify_sources()
    return count, digest.hexdigest()


def _verified_validation(path, label, checkpoint_sha, identity, frame_count, frame_digest):
    """Require a completed matching full-validation selection before test."""
    path = Path(path).expanduser().resolve(strict=True)
    before = _stat(path)
    if before[0] > 4 * 1024 * 1024:
        raise ValueError(f"{label}: validation report exceeds size limit")
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024 or _stat(path) != before:
        raise ValueError(f"{label}: validation report changed while reading")
    try:
        report = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{label}: invalid validation report") from exc
    if not isinstance(report, dict):
        raise ValueError(f"{label}: validation report must be an object")
    if (report.get("status") != "complete" or report.get("kind") != label or
            report.get("channel") != CHANNEL or report.get("split") != "val" or
            report.get("test_split_read") is not False):
        raise ValueError(f"{label}: requires a completed {label} validation on sog_candidates/val")
    if (report.get("condition") != [0., 1.] or report.get("index_identity") != identity or
            report.get("frames_per_checkpoint") != frame_count or
            report.get("frame_identity_sha256") != frame_digest):
        raise ValueError(f"{label}: validation condition, bound index or validation frames differ")
    if report.get("selected_checkpoint_sha256") != checkpoint_sha:
        raise ValueError(f"{label}: checkpoint is not the validation-selected checkpoint")
    candidates = report.get("candidates")
    selected = report.get("selected_index")
    if (not isinstance(candidates, list) or not candidates or isinstance(selected, bool) or
            not isinstance(selected, int) or not 0 <= selected < len(candidates)):
        raise ValueError(f"{label}: invalid validation candidate selection")
    errors = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("frames") != frame_count or candidate.get("frame_identity_sha256") != frame_digest:
            raise ValueError(f"{label}: validation candidate frame coverage differs")
        force = candidate.get("force")
        value = force.get("mae") if isinstance(force, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{label}: invalid validation force MAE")
        errors.append(value)
    if candidates[selected].get("checkpoint_sha256") != checkpoint_sha or selected != min(range(len(errors)), key=errors.__getitem__):
        raise ValueError(f"{label}: report does not select the minimum validation force MAE")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "selected_checkpoint_sha256": checkpoint_sha, "frames": frame_count,
            "frame_identity_sha256": frame_digest, "selection": "minimum val force MAE"}, before


def predict(database, dpa, sog, output, charge_spin, *, validation_reports=None, loader=None, progress_seconds=30):
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
        if not isinstance(validation_reports, dict) or set(validation_reports) != {"dpa", "sog"}:
            raise ValueError("completed DPA and SOG validation reports are required before test")
        state["checkpoints"] = {"dpa": {"path": str(dpa), "sha256": dpa_sha},
                                "sog": {"path": str(sog), "sha256": sog_sha}}
        loader = loader or _default_loader
        base_model, sog_model = loader(dpa), loader(sog)
        models = {"dpa": _model_details(base_model, "dpa"), "sog": _model_details(sog_model, "sog")}
        if set(models["dpa"]["type_map"]) != set(models["sog"]["type_map"]):
            raise ValueError("baseline and SOG checkpoint type maps contain different elements")
        state["architecture_comparison"] = _compare_architecture(models)
        state["models"] = models
        with IndexReader(database) as reader:
            total = reader.counts(CHANNEL, SPLIT)
            if total == 0:
                raise ValueError("the indexed sog_candidates/test split is empty")
            identity = _index_identity(reader.metadata)
            index_stat = _stat(reader.database)
            val_count, val_digest = _validation_frames(reader)
            validation = {}
            report_stats = {}
            for label, checkpoint_sha in (("dpa", dpa_sha), ("sog", sog_sha)):
                validation[label], report_stats[label] = _verified_validation(
                    validation_reports[label], label, checkpoint_sha, identity, val_count, val_digest)
            state.update(index_identity=identity, index_path=str(database), expected_frames=total,
                         validation_provenance=validation, fine_tune_parent_not_verified=True,
                         fine_tune_note="SOG initialization from the selected DPA checkpoint requires separate training provenance")
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
    parser.add_argument("--dpa", required=True, type=Path)
    parser.add_argument("--sog", required=True, type=Path)
    parser.add_argument("--dpa-validation", required=True, type=Path, help="completed DPA streaming val report")
    parser.add_argument("--sog-validation", required=True, type=Path, help="completed SOG streaming val report")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--charge-spin", nargs=2, type=float, required=True, metavar=("CHARGE", "MULTIPLICITY"))
    parser.add_argument("--progress-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        result = predict(args.database, args.dpa, args.sog, args.output, args.charge_spin,
                         validation_reports={"dpa": args.dpa_validation, "sog": args.sog_validation},
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
