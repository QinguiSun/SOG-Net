"""Validate a conversion's frame manifest against a complete, bound index.

Rebinding legitimately changes index metadata identity. This command compares
each frame's stable identifiers rather than requiring metadata hashes to match.
It checks manifest/source alignment and permutation bijectivity, not numerical
contents of converted .npy files. Use a separate content-integrity check for
transferred converted files before training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
import time

from cace_sog_training.data import IndexReader, SPLITS

CHANNEL = "sog_candidates"
MAX_JSON_BYTES = 4 * 1024 * 1024
SYSTEM = re.compile(r"n([1-9][0-9]*)_([0-9a-f]{24})\Z")
SET = re.compile(r"set\.(?:[0-9]{3}|[1-9][0-9]{3,})\Z")


def _stat(path):
    result = path.stat()
    return result.st_size, result.st_mtime_ns


def _identity(metadata):
    raw = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be a lowercase SHA256")
    return value


def _read_status(path):
    before = _stat(path)
    if before[0] > MAX_JSON_BYTES:
        raise ValueError("conversion status exceeds the size limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES or _stat(path) != before:
        raise ValueError("conversion status changed or exceeds the size limit")
    try:
        status = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid conversion status JSON") from exc
    if not isinstance(status, dict) or status.get("status") != "complete" or status.get("channel") != CHANNEL:
        raise ValueError("conversion status must be complete and belong to sog_candidates")
    if status.get("frames_manifest") != "frames.jsonl":
        raise ValueError("conversion status must identify the canonical frames.jsonl")
    if status.get("no_sampling") is not True or status.get("original_splits_preserved") is not True:
        raise ValueError("conversion status must retain all selected frames and existing splits")
    counts = status.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(SPLITS):
        raise ValueError("conversion status has invalid split counts")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in counts.values()):
        raise ValueError("conversion status split counts must be positive integers")
    total = sum(counts.values())
    for key in ("indexed_frames", "exported_frames"):
        if isinstance(status.get(key), bool) or not isinstance(status.get(key), int) or status.get(key) != total:
            raise ValueError("conversion status " + key + " differs from split counts")
    _sha(status.get("source_index_metadata_sha256"), "source index metadata identity")
    return status, before, hashlib.sha256(raw).hexdigest()


def _integer(value, field, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > 2**63 - 2:
        raise ValueError("invalid " + field)
    return value


def _permutation(value, natoms):
    if not isinstance(value, list) or len(value) != natoms:
        raise ValueError("canonical_to_source length differs from the indexed atom count")
    seen = bytearray(natoms)
    for index in value:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= natoms or seen[index]:
            raise ValueError("canonical_to_source is not a bijective atom permutation")
        seen[index] = 1


def _system_identity(source, natoms):
    """Derive the conversion's system name from indexed atom counts only."""
    counts = json.loads(source["counts_json"])
    if not isinstance(counts, dict) or not counts:
        raise ValueError("bound index has invalid elemental counts")
    parsed = []
    for key, count in counts.items():
        if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
            raise ValueError("bound index has invalid atomic number")
        z = int(key)
        if key != str(z) or not 1 <= z <= 118 or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("bound index has invalid atomic number or count")
        parsed.append((z, count))
    if sum(count for _, count in parsed) != natoms:
        raise ValueError("bound index atom count and elemental counts differ")
    digest = hashlib.sha256()
    for z, count in sorted(parsed):
        digest.update(z.to_bytes(2, "little") * count)
    return "n%d_%s" % (natoms, digest.hexdigest()[:24])


def _publish_json(path, value):
    """Publish a new completed report atomically, without overwriting."""
    handle, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    temp = Path(temporary)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(temp), str(path))
        except FileExistsError as exc:
            raise ValueError("output report already exists; choose a new path") from exc
        except OSError as exc:
            raise OSError("report filesystem must support atomic same-directory hard links") from exc
    finally:
        if temp.exists():
            temp.unlink()


def verify_alignment(database, converted, output, progress_seconds=30):
    """Compare every converted-frame manifest entry to the bound source index."""
    if (isinstance(progress_seconds, bool) or not isinstance(progress_seconds, (int, float))
            or not math.isfinite(progress_seconds) or progress_seconds <= 0):
        raise ValueError("progress_seconds must be positive")
    root = Path(converted).expanduser().resolve(strict=True)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError("output report already exists; choose a new path")
    if not root.is_dir():
        raise ValueError("converted must be a directory")
    status_path = root / "conversion_status.json"
    manifest_path = root / "frames.jsonl"
    for path in (status_path, manifest_path):
        try:
            path.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise ValueError("conversion metadata path escapes the converted directory") from exc
        if not path.is_file():
            raise ValueError("conversion metadata must be regular files")
    status, status_stat, status_sha = _read_status(status_path)
    manifest_stat = _stat(manifest_path)
    if not manifest_path.is_file():
        raise ValueError("conversion frames.jsonl must be a regular file")
    target.parent.mkdir(parents=True, exist_ok=True)

    with IndexReader(database) as reader:
        if CHANNEL not in reader.channels:
            raise ValueError("bound index does not contain sog_candidates")
        reader.verify_sources()
        index_stat = _stat(reader.database)
        bound_identity = _identity(reader.metadata)
        expected = {split: reader.counts(CHANNEL, split) for split in SPLITS}
        if status["counts"] != expected:
            raise ValueError("conversion and bound index have different split counts")

        with tempfile.TemporaryDirectory(prefix=".dpa4c-align-", dir=str(target.parent)) as temp:
            db = sqlite3.connect(str(Path(temp) / "seen.sqlite3"))
            try:
                db.execute("PRAGMA journal_mode=OFF")
                db.execute("PRAGMA synchronous=OFF")
                db.execute("PRAGMA temp_store=FILE")
                db.execute("PRAGMA cache_size=-8192")
                db.execute("PRAGMA mmap_size=0")
                db.execute("""CREATE TABLE seen(split TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    system TEXT NOT NULL, subset TEXT NOT NULL, row_no INTEGER NOT NULL,
                    PRIMARY KEY(split,ordinal), UNIQUE(split,system,subset,row_no)) WITHOUT ROWID""")
                counts = {split: 0 for split in SPLITS}
                digest = hashlib.sha256()
                line_no, last = 0, time.monotonic()
                with manifest_path.open("rb", buffering=1024 * 1024) as stream:
                    while True:
                        raw = stream.readline(MAX_JSON_BYTES + 1)
                        if not raw:
                            break
                        line_no += 1
                        if len(raw) > MAX_JSON_BYTES:
                            raise ValueError("conversion manifest line %d exceeds size limit" % line_no)
                        digest.update(raw)
                        try:
                            item = json.loads(raw.decode("utf-8"))
                            if not isinstance(item, dict):
                                raise ValueError("frame manifest entry is not an object")
                            split = item.get("split")
                            if item.get("channel") != CHANNEL or split not in SPLITS:
                                raise ValueError("invalid channel or original split")
                            ordinal = _integer(item.get("source_ordinal"), "source ordinal")
                            source_id = _integer(item.get("source_id"), "source record ID", 1)
                            line = _integer(item.get("manifest_line"), "original manifest line", 1)
                            source = reader.record(CHANNEL, split, ordinal)
                            if (source_id != source["id"] or line != source["manifest_line"] or
                                    item.get("component_id") != source["component_id"] or
                                    _sha(item.get("sha256"), "frame digest") != source["sha256"]):
                                raise ValueError("frame identifiers, hash or parent component differ from bound index")
                            natoms = int(source["natoms"])
                            if "natoms" in item and _integer(item["natoms"], "atom count", 1) != natoms:
                                raise ValueError("frame atom count differs from bound index")
                            _permutation(item.get("canonical_to_source"), natoms)
                            system, subset = item.get("system"), item.get("set")
                            match = SYSTEM.fullmatch(system) if isinstance(system, str) else None
                            if not match or match.group(1) != str(natoms):
                                raise ValueError("invalid canonical system path or atom count")
                            if system != _system_identity(source, natoms):
                                raise ValueError("converted system composition differs from bound index")
                            if not isinstance(subset, str) or not SET.fullmatch(subset):
                                raise ValueError("invalid canonical set path")
                            position = _integer(item.get("row"), "set row")
                            try:
                                db.execute("INSERT INTO seen VALUES (?,?,?,?,?)", (split, ordinal, system, subset, position))
                            except sqlite3.IntegrityError as exc:
                                raise ValueError("duplicate source frame or converted set row") from exc
                            counts[split] += 1
                        except (ValueError, TypeError, UnicodeError, IndexError) as exc:
                            raise ValueError("conversion manifest line %d: %s" % (line_no, exc)) from exc
                        if line_no % 10000 == 0:
                            db.commit()
                        if time.monotonic() - last >= progress_seconds:
                            print("Alignment: %s frames checked" % format(line_no, ","), file=sys.stderr, flush=True)
                            last = time.monotonic()
                db.commit()
                if counts != expected:
                    raise ValueError("converted manifest is missing or has extra frames")
                missing_row = db.execute("""SELECT 1 FROM seen GROUP BY split,system,subset
                    HAVING MIN(row_no) != 0 OR MAX(row_no) + 1 != COUNT(*) LIMIT 1""").fetchone()
                if missing_row:
                    raise ValueError("converted set manifest contains noncontiguous row numbers")
            finally:
                db.close()

        if _stat(manifest_path) != manifest_stat:
            raise ValueError("conversion manifest changed during alignment")
        _, status_stat_after, status_sha_after = _read_status(status_path)
        if status_stat_after != status_stat or status_sha_after != status_sha:
            raise ValueError("conversion status changed during alignment")
        if _stat(reader.database) != index_stat:
            raise ValueError("bound index changed during alignment")
        reader.verify_sources()
        report = {"status": "complete", "channel": CHANNEL, "counts": counts,
                  "frames_checked": sum(counts.values()), "source_index_metadata_sha256": status["source_index_metadata_sha256"],
                  "bound_index_metadata_sha256": bound_identity,
                  "bound_index_identity": bound_identity,
                  "metadata_identities_differ": status["source_index_metadata_sha256"] != bound_identity,
                  "conversion_status_sha256": status_sha, "frames_manifest_sha256": digest.hexdigest(),
                  "bound_database": str(reader.database), "converted_dir": str(root),
                  "verified": ["frame IDs, original channel/split, SHA256, parent component and manifest line",
                               "atom counts and bijectivity of recorded permutations",
                               "system name matches indexed elemental composition",
                               "unique source frames and unique contiguous set-row slots"],
                  "converted_npy_contents_verified": False,
                  "source_frame_bytes_rehashed": False,
                  "note": "Metadata identities may differ after sealing/rebinding; this compares indexed frame identities. Numerical array contents and atom-order semantics require separate checks."}
    _publish_json(target, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, type=Path, help="complete bound training index")
    parser.add_argument("--converted", required=True, type=Path, help="complete split-preserving conversion")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report path")
    args = parser.parse_args(argv)
    try:
        result = verify_alignment(args.database, args.converted, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (OSError, ValueError, sqlite3.Error) as exc:
        print("%s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
