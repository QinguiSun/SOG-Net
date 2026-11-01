"""Bounded element-stratified diagnostic pilot from the bound ELEMENTA index.

Samples are deliberately biased toward broad element coverage; they are not a
representative benchmark. Only train and validation records are read. This is
an engineering pilot, not a full-dataset conversion or a production trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from ase.data import chemical_symbols
from cace_sog_training.data import IndexReader
from .convert import TYPE_MAP, _check_condition, _key
from .index_smoke import (_append, _file_sha, _identity, _materialize,
                          _release, _write_json)

CHANNEL = "sog_candidates"


def select_records(records, per_element=4, seed=42, progress_seconds=30):
    """Select bounded bottom-k split components independently per element.

    The component hash is stable regardless of frame frequency; one frame per
    component/element is chosen by a second hash. At most 118 * per_element
    rows are retained. A frame selected for several elements appears once.
    """
    if isinstance(per_element, bool) or not isinstance(per_element, int) or not 1 <= per_element <= 32:
        raise ValueError("per-element must be between 1 and 32")
    buckets = {}
    last = time.monotonic()
    seen = 0
    for record in records:
        counts = json.loads(record["counts_json"])
        if not isinstance(counts, dict) or not counts:
            raise ValueError("invalid elemental counts")
        zs = sorted(int(z) for z in counts)
        if any(z < 1 or z > 118 or int(counts[str(z)]) <= 0 for z in zs):
            raise ValueError("invalid elemental counts")
        row = dict(record)
        identity = int(row["id"])
        component = str(row["component_id"])
        for z in zs:
            component_bytes = json.dumps([seed, z, component], ensure_ascii=True).encode("ascii")
            component_key = (hashlib.sha256(component_bytes).digest(), component)
            frame_key = (hashlib.sha256(f"{seed}:{identity}:{z}".encode("ascii")).digest(), identity)
            bucket = buckets.setdefault(z, {})
            previous = bucket.get(component)
            if previous is not None:
                if frame_key < previous[1]:
                    bucket[component] = (component_key, frame_key, row)
            elif len(bucket) < per_element:
                bucket[component] = (component_key, frame_key, row)
            else:
                worst = max(bucket, key=lambda key: bucket[key][0])
                if component_key < bucket[worst][0]:
                    del bucket[worst]
                    bucket[component] = (component_key, frame_key, row)
        seen += 1
        if progress_seconds and time.monotonic() - last >= progress_seconds:
            print(f"pilot selection: scanned {seen:,} indexed frames", flush=True)
            last = time.monotonic()
    selected = {int(entry[2]["id"]): entry[2] for bucket in buckets.values() for entry in bucket.values()}
    return sorted(selected.values(), key=lambda row: int(row["ordinal"])), sorted(buckets), seen


def prepare_pilot(database, release_config, output, *, per_element=4, seed=42, steps=200):
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100000:
        raise ValueError("steps must be between 1 and 100000")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32-1")
    database = Path(database).expanduser().resolve(strict=True)
    release_config = Path(release_config).expanduser().resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if os.path.lexists(output):
        raise ValueError("output already exists")
    output = output.resolve()
    config, config_sha, config_stat = _release(release_config)
    with IndexReader(database) as reader:
        source_root = Path(reader.metadata["config"]["export_dir"]).resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError("output must be outside the source export")
        identity = _identity(reader.metadata)
        index_stat = reader.database.stat()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        status = {"status": "partial", "debug_only": True, "no_training_executed": True,
                  "channel": CHANNEL, "test_split_read": False,
                  "selection": "per-element deterministic bottom-k components; biased, not representative",
                  "seed": seed, "per_element": per_element, "steps": steps,
                  "condition": [0, 1], "condition_is_not_measured_dft_state": True,
                  "source_index_metadata_sha256": identity, "release_config_sha256": config_sha}
        _write_json(output / "pilot_status.json", status)
        try:
            selected = {}
            coverage = {}
            for split in ("train", "val"):
                selected[split], coverage[split], scanned = select_records(
                    reader.iter_records((CHANNEL,), split), per_element, seed)
                if not selected[split]:
                    raise ValueError(f"{split} has no candidate frames")
                if scanned != reader.counts(CHANNEL, split):
                    raise ValueError(f"{split} count differs from index metadata")
                expected = set(reader.elements((CHANNEL,), split))
                if set(coverage[split]) != expected:
                    raise ValueError(f"{split} element coverage differs from metadata")
            train_components = {row["component_id"] for row in selected["train"]}
            val_components = {row["component_id"] for row in selected["val"]}
            if train_components & val_components:
                raise ValueError("a selected component appears in both train and val")
            groups = {}
            gram = np.zeros((118, 118), dtype=np.float64)
            with tempfile.TemporaryDirectory(prefix=".pilot-spool-", dir=str(output)) as directory:
                spool = Path(directory)
                manifest = output / "frames.jsonl"
                with manifest.open("w", encoding="utf8", newline="\n") as stream:
                    for split in ("train", "val"):
                        for record in selected[split]:
                            if record["channel"] != CHANNEL or record["split"] != split:
                                raise ValueError("record split/channel mismatch")
                            atoms = reader.read_atoms(record)
                            _check_condition(atoms.info)
                            order = np.argsort(atoms.numbers, kind="stable")
                            name, types = _key(atoms.numbers)
                            key = (split, name)
                            if key not in groups:
                                folder = spool / split / name
                                folder.mkdir(parents=True)
                                groups[key] = {"spool": folder, "types": types,
                                               "natoms": len(atoms), "rows": 0}
                            group = groups[key]
                            if group["types"] != types or group["natoms"] != len(atoms):
                                raise ValueError("composition hash collision")
                            for field, value in (("coord", atoms.positions[order]),
                                                 ("force", atoms.arrays["REF_forces"][order]),
                                                 ("box", atoms.cell.array),
                                                 ("energy", [atoms.info["REF_energy"]])):
                                _append(group["spool"] / (field + ".raw"),
                                        np.asarray(value, dtype="<f8").tobytes())
                            if split == "train":
                                x = np.bincount(atoms.numbers, minlength=119)[1:119].astype(float) / len(atoms)
                                gram += np.outer(x, x)
                            entry = {"channel": CHANNEL, "split": split, "source_id": int(record["id"]),
                                     "source_ordinal": int(record["ordinal"]), "frame_sha256": record["sha256"],
                                     "component_id": record["component_id"], "natoms": len(atoms),
                                     "system": name, "set": "set.000", "row": group["rows"],
                                     "canonical_to_source": order.tolist()}
                            stream.write(json.dumps(entry, separators=(",", ":"), allow_nan=False) + "\n")
                            group["rows"] += 1
                files = [manifest]
                for (split, name), group in groups.items():
                    folder = output / split / name
                    subset = folder / "set.000"
                    subset.mkdir(parents=True)
                    types_path, map_path = folder / "type.raw", folder / "type_map.raw"
                    types_path.write_text(" ".join(map(str, json.loads(group["types"]))) + "\n", encoding="ascii")
                    map_path.write_text(" ".join(TYPE_MAP) + "\n", encoding="ascii")
                    files.extend((types_path, map_path))
                    for field, width in (("coord", group["natoms"] * 3), ("force", group["natoms"] * 3),
                                         ("box", 9), ("energy", 1)):
                        path = subset / (field + ".npy")
                        _materialize(group["spool"] / (field + ".raw"), path, group["rows"], width)
                        files.append(path)
            reader.verify_sources()
            current = reader.database.stat()
            if _identity(reader.metadata) != identity or (current.st_size, current.st_mtime_ns) != (index_stat.st_size, index_stat.st_mtime_ns):
                raise ValueError("source index changed during pilot")
            current_release = release_config.stat()
            if (current_release.st_size, current_release.st_mtime_ns) != config_stat or _file_sha(release_config) != config_sha:
                raise ValueError("release configuration changed")
            training = config.setdefault("training", {})
            for field in ("stat_file", "stat_file_mode", "numb_epoch", "numb_epochs", "num_epoch", "num_epochs",
                          "numb_step", "numb_steps", "num_step", "num_steps", "stop_batch", "zero_stage"):
                training.pop(field, None)
            training.update({"training_data": {"systems": str(output / "train"), "batch_size": 1,
                                                "auto_prob_style": "prob_sys_size"},
                             "validation_data": {"systems": str(output / "val"), "batch_size": 1, "numb_batch": 1},
                             "num_steps": steps, "seed": seed, "enable_compile": False,
                             "save_freq": min(steps, 2000), "max_ckpt_keep": 3, "enable_ema": False})
            learning = config.setdefault("learning_rate", {})
            learning["start_lr"] = 1e-4
            if "stop_lr" in learning:
                learning["stop_lr"] = min(float(learning["stop_lr"]), 1e-5)
            config["validating"] = {"full_validation": False, "ema_full_validation": False}
            config_path = output / "input_dpa4c_pilot.json"
            _write_json(config_path, config)
            files.append(config_path)
            inventory = {p.relative_to(output).as_posix(): {"size": p.stat().st_size, "sha256": _file_sha(p)} for p in files}
            inventory_path = output / "pilot_files.json"
            _write_json(inventory_path, {"format": "dpa4c-stratified-pilot-files-v1", "files": inventory})
            active = [z - 1 for z in coverage["train"]]
            rank = int(np.linalg.matrix_rank(gram[np.ix_(active, active)]))
            status.update(status="complete", counts={s: len(selected[s]) for s in selected},
                          elements={s: [chemical_symbols[z] for z in coverage[s]] for s in coverage},
                          train_composition_rank=rank, train_element_count=len(active),
                          rank_note="rank of selected train composition matrix; no energy alignment is applied",
                          frames_manifest_sha256=_file_sha(manifest),
                          config_sha256=_file_sha(config_path), inventory_sha256=_file_sha(inventory_path))
            _write_json(output / "pilot_status.json", status)
            return status
        except BaseException as exc:
            status.update(status="partial", error=f"{type(exc).__name__}: {exc}"[:4096])
            _write_json(output / "pilot_status.json", status)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-element", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--assume-charge-spin", type=int, nargs=2, required=True)
    args = parser.parse_args()
    if args.assume_charge_spin != [0, 1]:
        parser.error("explicit --assume-charge-spin 0 1 is required")
    print(json.dumps(prepare_pilot(args.database, args.release_config, args.output,
                                  per_element=args.per_element, seed=args.seed, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
