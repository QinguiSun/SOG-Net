"""Create a tiny DPA4C smoke dataset directly from a complete source index.

Run this on the same machine and absolute data path on which the generated
config will be used, for example on an allocated SCNet GPU node after binding
the index. Only the first requested train/val frames are copied; no test
frames are read. This workspace is for interface checks, never final training
or benchmarking. The source export, index and release configuration are not
modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from ase.data import chemical_symbols

from cace_sog_training.data import IndexReader
from .convert import TYPE_MAP, _check_condition, _key

CHANNEL = "sog_candidates"
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_FRAMES_PER_SPLIT = 64


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path, value):
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _release(path):
    before = path.stat()
    if before.st_size > MAX_CONFIG_BYTES:
        raise ValueError("release configuration exceeds the bounded size limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    after = path.stat()
    if len(raw) > MAX_CONFIG_BYTES or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("release configuration changed or exceeded its size limit")
    config = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("release configuration must include a model object")
    model = config["model"]
    descriptor = model.get("descriptor", {})
    if model.get("type_map") != chemical_symbols[1:119] or descriptor.get("type") != "dpa4c":
        raise ValueError("release must contain the full H–Og DPA4C model section")
    if descriptor.get("add_chg_spin_ebd") is not True or descriptor.get("default_chg_spin") != [0, 1]:
        raise ValueError("release must use the reviewed default charge/spin condition [0,1]")
    if "long_range" in model:
        raise ValueError("release must be the short-range DPA4C baseline")
    return config, hashlib.sha256(raw).hexdigest(), (before.st_size, before.st_mtime_ns)


def _append(path, raw):
    with path.open("ab") as stream:
        stream.write(raw)


def _materialize(raw_path, target, rows, width):
    """Write one NPY at a time and retain at most one frame of data."""
    shape = (rows,) if width == 1 else (rows, width)
    array = np.lib.format.open_memmap(target, mode="w+", dtype="<f8", shape=shape)
    try:
        with raw_path.open("rb") as source:
            for row in range(rows):
                raw = source.read(width * 8)
                if len(raw) != width * 8:
                    raise ValueError("incomplete smoke data spool")
                value = np.frombuffer(raw, dtype="<f8", count=width)
                array[row] = value[0] if width == 1 else value
            if source.read(1):
                raise ValueError("extra bytes in smoke data spool")
        array.flush()
    finally:
        # Explicitly release the mapping, including on Windows where open
        # mappings can prevent temporary-file cleanup.
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
        del array


def prepare_index_smoke(database, release_config, output, assume_charge_spin,
                        frames_per_split=4, steps=200):
    """Prepare a small two-split workspace; never launch training."""
    if tuple(assume_charge_spin) != (0, 1):
        raise ValueError("explicit --assume-charge-spin 0 1 is required")
    if isinstance(frames_per_split, bool) or not isinstance(frames_per_split, int) or not 1 <= frames_per_split <= MAX_FRAMES_PER_SPLIT:
        raise ValueError(f"frames_per_split must be an integer from 1 to {MAX_FRAMES_PER_SPLIT}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    database = Path(database).expanduser().resolve(strict=True)
    source_config = Path(release_config).expanduser().resolve(strict=True)
    candidate = Path(output).expanduser().absolute()
    if os.path.lexists(candidate):
        raise ValueError("output already exists; choose a NEW directory")
    target = candidate.resolve()
    config, release_sha, release_stat = _release(source_config)
    with IndexReader(database) as reader:
        if CHANNEL not in reader.channels:
            raise ValueError("complete index does not include sog_candidates")
        source_root = Path(reader.metadata["config"]["export_dir"]).resolve()
        if target == source_root or source_root in target.parents:
            raise ValueError("smoke output must be outside the source export")
        counts = {split: min(frames_per_split, reader.counts(CHANNEL, split)) for split in ("train", "val")}
        if not all(counts.values()):
            raise ValueError("both train and val must contain indexed frames")
        index_identity = _identity(reader.metadata)
        index_stat = reader.database.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        status = {"status": "partial", "debug_only": True, "no_training_executed": True,
                  "channel": CHANNEL, "source_database": str(database), "source_index_metadata_sha256": index_identity,
                  "release_config": str(source_config), "release_config_sha256": release_sha,
                  "assumed_charge_spin": [0, 1], "condition_is_not_measured_dft_state": True,
                  "requested_frames_per_split": frames_per_split, "counts": counts,
                  "test_split_read": False, "full_dataset_modified": False,
                  "energy_unit": "eV total", "force_unit": "eV/angstrom",
                  "config_paths_are_machine_specific": True,
                  "requires_generation_on_training_machine": True,
                  "scope_note": "small deterministic debugging subset; not for final training or evaluation",
                  "source_frames": []}
        _write_json(target / "smoke_status.json", status)
        try:
            groups = {}
            with tempfile.TemporaryDirectory(prefix=".index-smoke-spool-", dir=str(target)) as directory:
                spool = Path(directory)
                manifest = target / "frames.jsonl"
                with manifest.open("w", encoding="utf-8", newline="\n") as stream:
                    for split in ("train", "val"):
                        written = 0
                        for record in reader.iter_records((CHANNEL,), split):
                            if written >= counts[split]:
                                break
                            if record["split"] != split or record["channel"] != CHANNEL or record["ordinal"] != written:
                                raise ValueError("indexed source frames are not in contiguous split order")
                            atoms = reader.read_atoms(record)  # verifies original frame bytes and labels
                            _check_condition(atoms.info)
                            if len(atoms) != record["natoms"]:
                                raise ValueError("source atom count differs from index")
                            order = np.argsort(atoms.numbers, kind="stable")
                            name, types = _key(atoms.numbers)
                            group_key = (split, name)
                            group = groups.get(group_key)
                            if group is None:
                                folder = spool / split / name
                                folder.mkdir(parents=True)
                                group = {"split": split, "system": name, "types": types,
                                         "natoms": len(atoms), "rows": 0, "spool": folder}
                                groups[group_key] = group
                            elif group["types"] != types or group["natoms"] != len(atoms):
                                raise ValueError("composition hash collision in smoke sample")
                            for name_of_array, value in (
                                ("coord", atoms.positions[order]),
                                ("force", atoms.arrays["REF_forces"][order]),
                                ("box", atoms.cell.array),
                                ("energy", [atoms.info["REF_energy"]]),
                            ):
                                _append(group["spool"] / (name_of_array + ".raw"),
                                        np.asarray(value, dtype="<f8").tobytes())
                            entry = {"channel": CHANNEL, "split": split, "source_id": int(record["id"]),
                                     "source_ordinal": int(record["ordinal"]), "frame_sha256": record["sha256"],
                                     "component_id": record["component_id"], "natoms": len(atoms),
                                     "system": name, "set": "set.000", "row": group["rows"],
                                     "canonical_to_source": order.tolist()}
                            stream.write(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n")
                            status["source_frames"].append({k: entry[k] for k in
                                ("split", "source_id", "source_ordinal", "frame_sha256")})
                            group["rows"] += 1
                            written += 1
                        if written != counts[split]:
                            raise ValueError("complete index has fewer source frames than its metadata")
                files = []
                for group in groups.values():
                    folder = target / group["split"] / group["system"]
                    subset = folder / "set.000"
                    subset.mkdir(parents=True)
                    types_path = folder / "type.raw"
                    map_path = folder / "type_map.raw"
                    types_path.write_text(" ".join(map(str, json.loads(group["types"]))) + "\n", encoding="ascii")
                    map_path.write_text(" ".join(TYPE_MAP) + "\n", encoding="ascii")
                    files.extend((types_path, map_path))
                    for name_of_array, width in (("coord", group["natoms"] * 3), ("force", group["natoms"] * 3),
                                                 ("box", 9), ("energy", 1)):
                        path = subset / (name_of_array + ".npy")
                        _materialize(group["spool"] / (name_of_array + ".raw"), path, group["rows"], width)
                        files.append(path)
            reader.verify_sources()
            new_stat = reader.database.stat()
            if _identity(reader.metadata) != index_identity or (new_stat.st_size, new_stat.st_mtime_ns) != (index_stat.st_size, index_stat.st_mtime_ns):
                raise ValueError("source index changed during smoke preparation")
            config_stat = source_config.stat()
            if (config_stat.st_size, config_stat.st_mtime_ns) != release_stat or _file_sha(source_config) != release_sha:
                raise ValueError("release configuration changed during smoke preparation")
            smoke = copy.deepcopy(config)
            training = smoke.setdefault("training", {})
            for field in ("stat_file", "stat_file_mode", "numb_epoch", "numb_epochs", "num_epoch", "num_epochs",
                          "numb_step", "numb_steps", "num_step", "num_steps", "stop_batch", "zero_stage"):
                training.pop(field, None)
            training.update({"training_data": {"systems": str(target / "train"), "batch_size": 1,
                                                "auto_prob_style": "prob_sys_size"},
                             "validation_data": {"systems": str(target / "val"), "batch_size": 1, "numb_batch": 1},
                             "num_steps": steps, "seed": 42, "enable_compile": False,
                             "save_freq": min(steps, 2000), "max_ckpt_keep": 3, "enable_ema": False})
            learning = smoke.setdefault("learning_rate", {})
            learning["start_lr"] = 1e-4
            if "stop_lr" in learning:
                learning["stop_lr"] = min(float(learning["stop_lr"]), 1e-5)
            smoke["validating"] = {"full_validation": False, "ema_full_validation": False}
            config_path = target / "input_dpa4c_smoke.json"
            _write_json(config_path, smoke)
            files.extend((manifest, config_path))
            inventory = {path.relative_to(target).as_posix(): {"size": path.stat().st_size, "sha256": _file_sha(path)}
                         for path in files}
            inventory_path = target / "smoke_files.json"
            _write_json(inventory_path, {"format": "dpa4c-index-smoke-files-v1", "files": inventory})
            status.update(status="complete", frames_manifest="frames.jsonl", frames_manifest_sha256=_file_sha(manifest),
                          data_inventory="smoke_files.json", data_inventory_sha256=_file_sha(inventory_path),
                          config=str(config_path), config_sha256=_file_sha(config_path),
                          systems_per_split={split: sum(key[0] == split for key in groups) for split in ("train", "val")},
                          steps=steps)
            _write_json(target / "smoke_status.json", status)
            return status
        except BaseException as exc:
            status.update(status="partial", error=f"{type(exc).__name__}: {exc}"[:4096])
            try:
                _write_json(target / "smoke_status.json", status)
            except OSError:
                pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--release-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assume-charge-spin", required=True, nargs=2, type=int, metavar=("CHARGE", "MULTIPLICITY"))
    parser.add_argument("--frames-per-split", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        result = prepare_index_smoke(args.database, args.release_config, args.output,
                                     args.assume_charge_spin, args.frames_per_split, args.steps)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; smoke workspace is partial.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
