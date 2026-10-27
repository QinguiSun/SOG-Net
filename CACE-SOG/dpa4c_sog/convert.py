"""Convert a complete ELEMENTA index to split-preserving DeePMD NPY systems.

Only sog_candidates is used.  The [0, 1] charge/multiplicity input is an
explicit model-condition assumption, never a DFT label.  Source frames are
read once with the index's SHA and E/F checks; a local SQLite spool groups
canonical atom orders without retaining a corpus in RAM.  Existing output
directories are never overwritten and a failed conversion remains partial.

The output uses standard DeePMD systems: type.raw, type_map.raw and per-set
coord/box/energy/force arrays.  Full mapping is in frames.jsonl, including a
canonical-to-source atom permutation for recovering original force ordering.
Units are eV, angstrom and eV/angstrom; energy is total energy, not per atom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
from ase.data import chemical_symbols

from cace_sog_training.data import IndexReader, SPLITS

CHANNEL = "sog_candidates"
TYPE_MAP = tuple(chemical_symbols[1:119])
if len(TYPE_MAP) != 118 or TYPE_MAP[0] != "H" or TYPE_MAP[-1] != "Og":
    raise RuntimeError("ASE periodic table does not match the released DPA4C type map")


def _write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _check_condition(info):
    """Reject metadata conflicting with the asserted neutral-singlet input."""
    for key in ("charge_spin", "charge_multiplicity"):
        if key in info:
            try:
                value = np.asarray(info[key], dtype=float).reshape(-1)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid explicit frame charge/spin condition: " + key) from exc
            if value.shape != (2,) or not np.isfinite(value).all() or not np.allclose(value, [0., 1.], atol=1e-10, rtol=0):
                raise ValueError("frame charge/spin condition conflicts with requested [0,1]: " + key)
    for key in ("spin_multiplicity", "multiplicity"):
        if key in info:
            try:
                value = np.asarray(info[key], dtype=float)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid explicit spin multiplicity: " + key) from exc
            if value.ndim != 0 or not np.isfinite(value) or abs(float(value) - 1.) > 1e-10:
                raise ValueError("frame multiplicity conflicts with requested [0,1]: " + key)


def _key(numbers):
    # A system holds a fixed ordered type sequence. The hash controls path
    # length; the complete sequence in SQLite detects even a hash collision.
    canonical = np.sort(np.asarray(numbers, dtype=np.int16), kind="stable")
    name = "n%d_%s" % (len(canonical), hashlib.sha256(canonical.astype("<u2").tobytes()).hexdigest()[:24])
    return name, json.dumps((canonical.astype(int) - 1).tolist(), separators=(",", ":"))


def _progress(stage, count, last):
    now = time.monotonic()
    if now - last >= 30:
        print(f"{stage}: {count:,} frames", file=sys.stderr, flush=True)
        return now
    return last


def convert(database, output, assume_charge_spin, set_frames=1024, set_bytes=64 * 1024 * 1024, commit_every=1000):
    """Convert all selected source frames without resplitting or deduplication.

    ``set_bytes`` caps numerical arrays in each chunk except when a single
    accepted frame itself exceeds it. The SQLite spool is disk-backed; its
    temporary disk use and the final data files can each be large.
    """
    if tuple(assume_charge_spin) != (0, 1):
        raise ValueError("explicit --assume-charge-spin 0 1 is required for the released OMol model default")
    if isinstance(set_frames, bool) or not isinstance(set_frames, int) or set_frames < 1:
        raise ValueError("set_frames must be a positive integer")
    if isinstance(set_bytes, bool) or not isinstance(set_bytes, int) or set_bytes < 1:
        raise ValueError("set_bytes must be a positive integer")
    if isinstance(commit_every, bool) or not isinstance(commit_every, int) or commit_every < 1:
        raise ValueError("commit_every must be a positive integer")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("output directory already exists; choose a new directory")
    with IndexReader(database) as reader:
        if CHANNEL not in reader.channels:
            raise ValueError("source index does not include sog_candidates")
        source_root = Path(reader.metadata["config"]["export_dir"]).resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError("conversion output must not be inside the source export directory")
        counts = {split: reader.counts(CHANNEL, split) for split in SPLITS}
        if any(counts[split] == 0 for split in SPLITS):
            raise ValueError("all train/val/test splits must contain indexed sog_candidates frames")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        status = {"status": "partial", "source_database": str(reader.database), "channel": CHANNEL,
                  "counts": counts, "assumed_charge_spin": [0, 1],
                  "charge_spin_is_model_input_assumption_not_verified_dft_state": True,
                  "output_energy_unit": "eV total", "output_force_unit": "eV/angstrom",
                  "no_sampling": True, "original_splits_preserved": True}
        _write_json(output / "conversion_status.json", status)
        spool = output / ".conversion.sqlite3"
        db = sqlite3.connect(str(spool))
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
            CREATE TABLE systems(split TEXT NOT NULL, system TEXT NOT NULL, natoms INTEGER NOT NULL,
                types TEXT NOT NULL, nframes INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(split, system));
            CREATE TABLE frames(id INTEGER PRIMARY KEY, split TEXT NOT NULL, system TEXT NOT NULL,
                source_id INTEGER NOT NULL, source_ordinal INTEGER NOT NULL, manifest_line INTEGER NOT NULL,
                component_id TEXT NOT NULL, sha256 TEXT NOT NULL, energy REAL NOT NULL,
                box BLOB NOT NULL, coord BLOB NOT NULL, force BLOB NOT NULL, permutation BLOB NOT NULL);
        """)
        try:
            seen, last = 0, time.monotonic()
            for split in SPLITS:
                for row in reader.iter_records((CHANNEL,), split):
                    atoms = reader.read_atoms(row)
                    _check_condition(atoms.info)
                    order = np.argsort(atoms.numbers, kind="stable")
                    name, types = _key(atoms.numbers)
                    natoms = len(atoms)
                    previous = db.execute("SELECT natoms,types FROM systems WHERE split=? AND system=?", (split, name)).fetchone()
                    if previous is None:
                        db.execute("INSERT INTO systems VALUES(?,?,?,?,0)", (split, name, natoms, types))
                    elif previous != (natoms, types):
                        raise ValueError("system identity collision")
                    db.execute("UPDATE systems SET nframes=nframes+1 WHERE split=? AND system=?", (split, name))
                    db.execute("INSERT INTO frames(split,system,source_id,source_ordinal,manifest_line,component_id,sha256,energy,box,coord,force,permutation) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (split, name, int(row["id"]), int(row["ordinal"]), int(row["manifest_line"]), row["component_id"], row["sha256"],
                         float(atoms.info["REF_energy"]), np.asarray(atoms.cell, dtype="<f8").tobytes(),
                         np.asarray(atoms.positions[order], dtype="<f8").tobytes(),
                         np.asarray(atoms.arrays["REF_forces"][order], dtype="<f8").tobytes(),
                         np.asarray(order, dtype="<u4").tobytes()))
                    seen += 1
                    if seen % commit_every == 0:
                        db.commit()
                    last = _progress("Spooling", seen, last)
            reader.verify_sources()
            if seen != sum(counts.values()):
                raise ValueError("indexed frame count changed during conversion")
            db.commit()
            db.execute("CREATE INDEX frame_groups ON frames(split,system,id)")
            db.commit()

            exported, last = 0, time.monotonic()
            with (output / "frames.jsonl").open("w", encoding="utf-8", newline="\n") as manifest, \
                 (output / "systems.jsonl").open("w", encoding="utf-8", newline="\n") as systems:
                groups = db.execute("SELECT split,system,natoms,types,nframes FROM systems ORDER BY split,system")
                for split, name, natoms, types_json, nframes in groups:
                    folder = output / split / name
                    folder.mkdir(parents=True)
                    types = json.loads(types_json)
                    (folder / "type.raw").write_text(" ".join(map(str, types)) + "\n", encoding="ascii")
                    (folder / "type_map.raw").write_text(" ".join(TYPE_MAP) + "\n", encoding="ascii")
                    systems.write(json.dumps({"split": split, "system": name, "frames": nframes, "natoms": natoms}) + "\n")
                    # Four float64 arrays: coord, force, box and energy.
                    bytes_per_frame = (6 * natoms + 10) * 8
                    chunk = min(set_frames, max(1, set_bytes // bytes_per_frame))
                    frames = db.execute("SELECT source_id,source_ordinal,manifest_line,component_id,sha256,energy,box,coord,force,permutation FROM frames WHERE split=? AND system=? ORDER BY id", (split, name))
                    offset = 0
                    set_id = 0
                    while offset < nframes:
                        length = min(chunk, nframes - offset)
                        subset = folder / ("set.%03d" % set_id)
                        subset.mkdir()
                        arrays = {
                            "coord": np.lib.format.open_memmap(subset / "coord.npy", mode="w+", dtype="<f8", shape=(length, natoms * 3)),
                            "force": np.lib.format.open_memmap(subset / "force.npy", mode="w+", dtype="<f8", shape=(length, natoms * 3)),
                            "box": np.lib.format.open_memmap(subset / "box.npy", mode="w+", dtype="<f8", shape=(length, 9)),
                            "energy": np.lib.format.open_memmap(subset / "energy.npy", mode="w+", dtype="<f8", shape=(length,)),
                        }
                        try:
                            for pos in range(length):
                                record = frames.fetchone()
                                if record is None:
                                    raise ValueError("spooled system has fewer rows than its count")
                                source_id, ordinal, manifest_line, component, sha, energy, box, coords, force, order = record
                                arrays["energy"][pos] = energy
                                arrays["box"][pos] = np.frombuffer(box, dtype="<f8", count=9)
                                arrays["coord"][pos] = np.frombuffer(coords, dtype="<f8", count=natoms * 3)
                                arrays["force"][pos] = np.frombuffer(force, dtype="<f8", count=natoms * 3)
                                manifest.write(json.dumps({"source_id": source_id, "source_ordinal": ordinal,
                                    "manifest_line": manifest_line, "component_id": component, "sha256": sha,
                                    "channel": CHANNEL, "split": split, "system": name,
                                    "set": subset.name, "row": pos,
                                    "canonical_to_source": np.frombuffer(order, dtype="<u4", count=natoms).tolist()},
                                    separators=(",", ":")) + "\n")
                                exported += 1
                                last = _progress("Writing DeePMD", exported, last)
                            for array in arrays.values():
                                array.flush()
                        finally:
                            arrays.clear()
                        offset += length
                        set_id += 1
                    if frames.fetchone() is not None:
                        raise ValueError("spooled system has more rows than its count")
            reader.verify_sources()
            if exported != seen:
                raise ValueError("conversion manifest and indexed frame counts differ")
            status.update(status="complete", indexed_frames=seen, exported_frames=exported,
                          type_map=list(TYPE_MAP), frames_manifest="frames.jsonl", systems_manifest="systems.jsonl",
                          grouping="split, actual atom count, sorted element types; positions and forces reordered together",
                          dp_type_mapping="atomic_number minus one (H=0, ..., Og=117)",
                          set_frame_cap=set_frames, set_byte_cap=set_bytes,
                          source_index_metadata_sha256=hashlib.sha256(json.dumps(reader.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest())
            db.close()
            spool.unlink()
            _write_json(output / "conversion_status.json", status)
            return status
        except BaseException as exc:
            db.close()
            status.update(status="partial", error=str(exc)[:4096])
            try:
                _write_json(output / "conversion_status.json", status)
            except OSError:
                pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assume-charge-spin", nargs=2, type=int, required=True, metavar=("CHARGE", "MULTIPLICITY"),
                        help="explicit model-input assumption, not a DFT state measurement; released OMol model requires 0 1")
    parser.add_argument("--set-frames", type=int, default=1024)
    parser.add_argument("--set-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--commit-every", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        result = convert(args.database, args.output, args.assume_charge_spin,
                         args.set_frames, args.set_bytes, args.commit_every)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted: conversion is partial; use a new output directory.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
