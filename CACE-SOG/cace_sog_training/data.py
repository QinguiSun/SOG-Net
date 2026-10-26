"""Disk-backed, one-frame-at-a-time access to an ELEMENTA export.

This module intentionally imports neither CACE nor PyTorch. Indexing can run on
a CPU node without loading the model stack. Existing channel/split assignments
are retained; indexing never samples frames or rewrites source extxyz files.
"""
from collections import Counter, OrderedDict
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sqlite3
import sys
import time

import numpy as np
from ase.io import read as ase_read

from . import DEFAULT_CHANNELS

FORMAT_VERSION = 1
CHANNELS = DEFAULT_CHANNELS + (
    "spin_review", "charge_review", "boundary_review", "electronic_review", "diagnostic_review"
)
SPLITS = ("train", "val", "test")
MAX_MANIFEST_LINE = 4 * 1024 * 1024
MAX_FRAME_BYTES = 64 * 1024 * 1024
MAX_ATOMS = 100000


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _connect(path, readonly=False):
    path = Path(path).expanduser().resolve()
    if readonly:
        db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    else:
        db = sqlite3.connect(str(path))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA cache_size=-32768")
    if not readonly:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
    return db


def _stat(path):
    info = Path(path).stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _sha256_file(path):
    """A full manifest digest, with fixed memory and infrequent progress."""
    digest = hashlib.sha256()
    size = 0
    last = time.monotonic()
    with Path(path).open("rb", buffering=8 * 1024 * 1024) as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            if time.monotonic() - last >= 30:
                print("Manifest integrity: hashed %.2f GiB" % (size / 1073741824), file=sys.stderr, flush=True)
                last = time.monotonic()
    return digest.hexdigest()


def _read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_LINE + 1)
    if len(raw) > MAX_MANIFEST_LINE:
        raise ValueError("oversized export status/summary: " + str(path))
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: " + str(path))
    return value


def _channels(values):
    if isinstance(values, str):
        values = (values,)
    result = tuple(dict.fromkeys(values))
    if not result:
        raise ValueError("select at least one channel")
    unknown = set(result) - set(CHANNELS)
    if unknown:
        raise ValueError("unsupported or unapproved channel(s): " + ", ".join(sorted(unknown)))
    return result


def _safe_export_path(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("missing export_path")
    windows = PureWindowsPath(relative)
    normalized = relative.replace("\\", "/")
    parts = normalized.split("/")
    if windows.drive or windows.root or normalized.startswith("/") or ".." in parts:
        raise ValueError("export_path must remain relative to export directory: " + relative)
    if any(part in ("", ".") for part in parts):
        raise ValueError("noncanonical export_path: " + relative)
    path = (root.joinpath(*parts)).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("export_path escapes export directory: " + relative) from exc
    if not path.is_file():
        raise ValueError("export_path is not a regular file: " + relative)
    return path


def _integer(value, name, minimum=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError("invalid " + name)
    return value


def _check_reference_metadata(atoms):
    """Enforce this adapter's neutral, ELEMENTA E/F reference policy.

    Missing charge/convergence metadata remain a source-protocol assumption;
    absence is not converted into a measured zero charge or converged label.
    Explicit conflicting or incompatible evidence must not be hidden by a
    higher-priority export channel such as spin_review.
    """
    values = []
    for key in ("net_charge", "total_charge", "charge"):
        if key not in atoms.info:
            continue
        raw = atoms.info[key]
        if isinstance(raw, (bool, np.bool_)) or np.asarray(raw).ndim != 0:
            raise ValueError("invalid explicit physical net charge: " + key)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid explicit physical net charge: " + key) from exc
        if not math.isfinite(value):
            raise ValueError("nonfinite explicit physical net charge: " + key)
        values.append((key, value))
    if values and any(abs(value - values[0][1]) > 1e-10 for _, value in values):
        raise ValueError("conflicting explicit physical net charge fields")
    if values and abs(values[0][1]) > 1e-10:
        raise ValueError("nonzero physical net charge requires a charge-conditioned model; this adapter uses a neutral-source policy")

    units = {
        "energy_unit": ("ev",),
        "force_unit": ("ev/angstrom", "ev/ang", "ev/å"),
        "length_unit": ("angstrom", "ang", "å"),
    }
    for key, allowed in units.items():
        if key in atoms.info and str(atoms.info[key]).strip().lower() not in allowed:
            raise ValueError("incompatible explicit " + key + ": " + str(atoms.info[key]))

    for key in ("electronic_converged", "scf_converged"):
        if key not in atoms.info:
            continue
        value = atoms.info[key]
        if isinstance(value, (bool, np.bool_)):
            false = not bool(value)
        elif isinstance(value, str):
            false = value.strip().lower() in ("f", "false", "0", "no", "n")
        elif np.asarray(value).ndim == 0:
            false = value == 0
        else:
            false = False  # Unknown provenance is not promoted to convergence.
        if false:
            raise ValueError("electronic nonconvergence explicitly reported: " + key)


def _parse_frame(raw, expected_natoms=None, require_periodic=True):
    if not raw or len(raw) > MAX_FRAME_BYTES:
        raise ValueError("empty or oversized frame")
    text = raw.decode("utf-8")
    lines = text.splitlines()
    try:
        natoms = int(lines[0])
    except (ValueError, IndexError) as exc:
        raise ValueError("invalid extxyz atom count") from exc
    if not 1 <= natoms <= MAX_ATOMS or len(lines) != natoms + 2:
        raise ValueError("frame atom count/length mismatch or parser guard exceeded")
    if expected_natoms is not None and natoms != expected_natoms:
        raise ValueError("manifest atom count differs from extxyz")
    atoms = ase_read(io.StringIO(text), format="extxyz", index=0)
    _check_reference_metadata(atoms)
    if len(atoms) != natoms or not np.all(atoms.numbers > 0):
        raise ValueError("invalid atom species/count")
    cell = np.asarray(atoms.cell, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (natoms, 3) or cell.shape != (3, 3) or not (
            np.isfinite(positions).all() and np.isfinite(cell).all()):
        raise ValueError("invalid/nonfinite positions or cell")
    determinant = float(np.linalg.det(cell))
    if not math.isfinite(determinant) or determinant <= 1e-8:
        raise ValueError("periodic SOG requires a nondegenerate, positive-determinant cell; source was not changed")
    if require_periodic and not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError("periodic SOG indexing requires three periodic boundary directions")
    results = atoms.calc.results if atoms.calc is not None and hasattr(atoms.calc, "results") else {}
    energy = atoms.info.get("energy")
    if energy is None:
        energy = results.get("energy")
    forces = atoms.arrays.get("forces")
    if forces is None:
        forces = results.get("forces")
    if energy is None or forces is None:
        raise ValueError("missing DFT energy or forces; no zero labels are synthesized")
    try:
        energy = float(energy)
        forces = np.asarray(forces, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid DFT energy/forces") from exc
    if not math.isfinite(energy) or forces.shape != (natoms, 3) or not np.isfinite(forces).all():
        raise ValueError("nonfinite or incorrectly shaped DFT energy/forces")
    atoms.info["REF_energy"] = energy
    atoms.set_array("REF_forces", forces.copy())
    return atoms


def _put(db, key, value):
    db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, _json(value)))


def _metadata(db):
    try:
        return {row[0]: json.loads(row[1]) for row in db.execute("SELECT key,value FROM metadata")}
    except (sqlite3.Error, ValueError) as exc:
        raise ValueError("not an ELEMENTA training index") from exc


def _schema(db):
    db.executescript("""
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
      CREATE TABLE files(id INTEGER PRIMARY KEY,path TEXT UNIQUE NOT NULL,size INTEGER NOT NULL,mtime_ns INTEGER NOT NULL);
      CREATE TABLE records(id INTEGER PRIMARY KEY,channel TEXT NOT NULL,split TEXT NOT NULL,
        ordinal INTEGER NOT NULL,file_id INTEGER NOT NULL,offset INTEGER NOT NULL,length INTEGER NOT NULL,
        sha256 TEXT NOT NULL,natoms INTEGER NOT NULL,energy REAL NOT NULL,counts_json TEXT NOT NULL,
        component_id TEXT NOT NULL,manifest_line INTEGER NOT NULL,
        UNIQUE(channel,split,ordinal));
      CREATE INDEX records_component ON records(component_id,split);
      CREATE TABLE component_splits(component_id TEXT PRIMARY KEY,split TEXT NOT NULL) WITHOUT ROWID;
    """)


def _export_status(root, channels):
    summary_path = root / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "complete":
        raise ValueError("only a complete export can be indexed")
    status_path = root / "run_status.json"
    if status_path.exists() and _read_json(status_path).get("status") != "complete":
        raise ValueError("export run_status is not complete")
    expected = {channel: {split: 0 for split in SPLITS} for channel in channels}
    seen = set()
    values = summary.get("counts_by_channel_and_split")
    if not isinstance(values, list):
        raise ValueError("export summary lacks channel/split counts")
    for item in values:
        channel, split = item.get("channel"), item.get("split")
        if channel not in channels:
            continue
        if split not in SPLITS or (channel, split) in seen:
            raise ValueError("invalid/duplicate channel count in export summary")
        seen.add((channel, split))
        expected[channel][split] = _integer(item.get("frames"), "summary frames")
    return expected, hashlib.sha256(summary_path.read_bytes()).hexdigest()


def _verify_files(db):
    for row in db.execute("SELECT path,size,mtime_ns FROM files"):
        current = _stat(row["path"])
        if current != {"size": row["size"], "mtime_ns": row["mtime_ns"]}:
            raise ValueError("indexed export file changed: " + row["path"])


def build_index(export_dir, database, channels=DEFAULT_CHANNELS, resume=False,
                commit_every=1000, limit_records=None):
    """Index selected manifest records with atomic checkpoints and bounded RAM.

    limit_records is a test-only per-invocation manifest-line limit: the index
    remains partial, and normal IndexReader construction rejects it. A resume
    must use the same root, channels and unchanged inputs. Metadata carries a
    full SHA256 of the manifest, so hashing it requires an extra sequential read.
    """
    channels = _channels(channels)
    _integer(commit_every, "commit_every", 1)
    if limit_records is not None:
        _integer(limit_records, "limit_records", 1)
    root = Path(export_dir).expanduser().resolve(strict=True)
    manifest = root / "manifest.jsonl"
    before = _stat(manifest)
    expected, summary_sha = _export_status(root, channels)
    manifest_sha = _sha256_file(manifest)
    if _stat(manifest) != before:
        raise ValueError("manifest changed during integrity hashing")
    config = {"format_version": FORMAT_VERSION, "export_dir": str(root), "manifest": str(manifest),
              "manifest_sha256": manifest_sha, "manifest_stat": before, "summary_sha256": summary_sha,
              "channels": list(channels), "periodic": True, "max_atoms": MAX_ATOMS,
              "max_frame_bytes": MAX_FRAME_BYTES, "max_manifest_line": MAX_MANIFEST_LINE,
              "reference_policy": {
                  "charge": "reject_explicit_nonzero_invalid_or_conflicting; missing_remains_source_assumption",
                  "units": "total_energy_eV_forces_eV_per_Angstrom_positions_Angstrom",
                  "convergence": "reject_any_explicit_false; missing_or_unknown_is_not_verified",
              }}
    database = Path(database).expanduser().resolve()
    existing = database.exists()
    if existing and not resume:
        raise ValueError("index exists; use --resume for unchanged inputs or choose a new database")
    saved_metadata = None
    if existing:
        # Refuse changed inputs before opening a writable connection or changing
        # journal settings on an existing index (including unrelated databases).
        inspection = _connect(database, readonly=True)
        try:
            saved_metadata = _metadata(inspection)
            if saved_metadata.get("config") != config:
                raise ValueError("index sources/settings changed; use original inputs or a new database")
            _verify_files(inspection)
            if saved_metadata.get("status") == "complete":
                return saved_metadata["summary"]
        finally:
            inspection.close()
    database.parent.mkdir(parents=True, exist_ok=True)
    db = _connect(database)
    handles = OrderedDict()
    may_write_metadata = False
    try:
        if existing:
            metadata = _metadata(db)
            if metadata != saved_metadata:
                raise ValueError("index changed after resume preflight; another writer may be active")
            if metadata.get("config") != config:
                raise ValueError("index sources/settings changed; use original inputs or a new database")
            _verify_files(db)
            if metadata.get("status") == "complete":
                return metadata["summary"]
            state = metadata["checkpoint"]
            may_write_metadata = True
        else:
            _schema(db)
            state = {"next_offset": 0, "line": 0, "indexed": 0, "skipped": 0,
                     "counts": {channel: {split: 0 for split in SPLITS} for channel in channels},
                     "elements": {channel: {split: [] for split in SPLITS} for channel in channels}}
            _put(db, "config", config)
            _put(db, "format_version", FORMAT_VERSION)
            _put(db, "manifest_sha256", manifest_sha)
            _put(db, "export_dir", str(root))
            _put(db, "channels", list(channels))
            _put(db, "checkpoint", state)
            _put(db, "status", "partial")
            db.commit()
            may_write_metadata = True
        element_sets = {channel: {split: set(state["elements"][channel][split]) for split in SPLITS}
                        for channel in channels}
        _put(db, "status", "partial")
        db.commit()
        run_lines = 0
        reached_eof = False
        last_progress = time.monotonic()

        def checkpoint(position):
            if _stat(manifest) != before:
                raise ValueError("manifest changed while indexing")
            _verify_files(db)
            state["next_offset"] = position
            state["elements"] = {channel: {split: sorted(element_sets[channel][split]) for split in SPLITS}
                                 for channel in channels}
            _put(db, "checkpoint", state)
            _put(db, "counts", state["counts"])
            _put(db, "elements", state["elements"])
            db.commit()

        with manifest.open("rb", buffering=4 * 1024 * 1024) as stream:
            stream.seek(state["next_offset"])
            while limit_records is None or run_lines < limit_records:
                raw_line = stream.readline(MAX_MANIFEST_LINE + 1)
                if not raw_line:
                    reached_eof = True
                    break
                if len(raw_line) > MAX_MANIFEST_LINE:
                    raise ValueError("manifest line exceeds bounded parser limit")
                state["line"] += 1
                run_lines += 1
                try:
                    item = json.loads(raw_line.decode("utf-8"))
                    if not isinstance(item, dict):
                        raise ValueError("manifest row must be an object")
                    channel, split = item.get("channel"), item.get("split")
                    if channel not in channels:
                        state["skipped"] += 1
                    else:
                        if split not in SPLITS:
                            raise ValueError("invalid split; original assignments are never regenerated")
                        path = _safe_export_path(root, item.get("export_path"))
                        offset = _integer(item.get("export_offset"), "export_offset")
                        length = _integer(item.get("export_length"), "export_length", 1)
                        natoms = _integer(item.get("natoms"), "natoms", 1)
                        if length > MAX_FRAME_BYTES or natoms > MAX_ATOMS:
                            raise ValueError("frame exceeds bounded parser limits")
                        sha = item.get("sha256")
                        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                            raise ValueError("invalid frame sha256")
                        component = item.get("component_id")
                        if not isinstance(component, str) or not component:
                            raise ValueError("missing component_id; cannot verify original split isolation")
                        file_row = db.execute("SELECT * FROM files WHERE path=?", (str(path),)).fetchone()
                        if file_row is None:
                            current = _stat(path)
                            cursor = db.execute("INSERT INTO files(path,size,mtime_ns) VALUES(?,?,?)",
                                                (str(path), current["size"], current["mtime_ns"]))
                            file_row = {"id": cursor.lastrowid, "size": current["size"], "mtime_ns": current["mtime_ns"]}
                        elif _stat(path) != {"size": file_row["size"], "mtime_ns": file_row["mtime_ns"]}:
                            raise ValueError("export file changed while indexing: " + str(path))
                        if offset + length > file_row["size"]:
                            raise ValueError("frame offset/length exceeds export file")
                        key = str(path)
                        handle = handles.pop(key, None)
                        if handle is None:
                            handle = path.open("rb", buffering=1024 * 1024)
                        handles[key] = handle
                        if len(handles) > 16:
                            handles.popitem(last=False)[1].close()
                        handle.seek(offset)
                        raw = handle.read(length)
                        if len(raw) != length or hashlib.sha256(raw).hexdigest() != sha:
                            raise ValueError("frame SHA256 mismatch")
                        atoms = _parse_frame(raw, natoms)
                        previous = db.execute("SELECT split FROM component_splits WHERE component_id=?", (component,)).fetchone()
                        if previous is not None and previous[0] != split:
                            raise ValueError("component_id crosses train/val/test: " + component)
                        if previous is None:
                            db.execute("INSERT INTO component_splits VALUES(?,?)", (component, split))
                        ordinal = state["counts"][channel][split]
                        counts = {str(int(z)): int(count) for z, count in Counter(atoms.numbers.tolist()).items()}
                        db.execute("""INSERT INTO records(channel,split,ordinal,file_id,offset,length,sha256,natoms,
                          energy,counts_json,component_id,manifest_line) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                                   (channel, split, ordinal, file_row["id"], offset, length, sha, natoms,
                                    atoms.info["REF_energy"], _json(counts), component, state["line"]))
                        state["counts"][channel][split] += 1
                        element_sets[channel][split].update(int(z) for z in counts)
                        state["indexed"] += 1
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    raise ValueError("manifest line %d: %s" % (state["line"], str(exc))) from exc
                if run_lines % commit_every == 0:
                    checkpoint(stream.tell())
                if time.monotonic() - last_progress >= 30:
                    print("Indexing: %s manifest rows; %s selected frames" %
                          (format(state["line"], ","), format(state["indexed"], ",")), file=sys.stderr, flush=True)
                    last_progress = time.monotonic()
            checkpoint(stream.tell())
        if reached_eof:
            if state["counts"] != expected:
                raise ValueError("indexed channel/split counts differ from complete export summary")
            status = "complete"
        else:
            status = "partial"
        result = {"status": status, "format_version": FORMAT_VERSION, "database": str(database),
                  "export_dir": str(root), "channels": list(channels), "counts": state["counts"],
                  "indexed_frames": state["indexed"], "manifest_rows": state["line"],
                  "manifest_sha256": manifest_sha, "no_sampling": True,
                  "split_policy": "preserved; component_id cannot cross splits"}
        _put(db, "summary", result)
        _put(db, "status", status)
        _put(db, "counts", state["counts"])
        _put(db, "elements", state["elements"])
        db.commit()
        return result
    except BaseException as exc:
        db.rollback()
        if may_write_metadata:
            _put(db, "status", "partial")
            _put(db, "last_error", str(exc)[:4096])
            db.commit()
        raise
    finally:
        for handle in handles.values():
            handle.close()
        db.close()


class IndexReader:
    """Per-process, read-only reader; never construct once and share across workers."""
    def __init__(self, database, require_complete=True):
        self.database = Path(database).expanduser().resolve(strict=True)
        self.db = _connect(self.database, readonly=True)
        self._handles = OrderedDict()
        self._pid = os.getpid()
        try:
            self.metadata = _metadata(self.db)
            if self.metadata.get("config", {}).get("format_version") != FORMAT_VERSION:
                raise ValueError("unsupported training index format")
            if require_complete and self.metadata.get("status") != "complete":
                raise ValueError("training/evaluation require a complete index, not a partial smoke-test index")
            self.channels = tuple(self.metadata["config"]["channels"])
            self.verify_sources()
        except BaseException:
            self.close()
            raise

    def _check_process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("create a separate IndexReader inside each DataLoader worker")

    def counts(self, channel, split):
        self._check_process()
        if channel not in self.channels or split not in SPLITS:
            raise ValueError("channel/split not present in this index")
        return int(self.metadata.get("counts", self.metadata["checkpoint"]["counts"])[channel][split])

    def elements(self, channels, split):
        self._check_process()
        channels = _channels(channels)
        if not set(channels).issubset(self.channels) or split not in SPLITS:
            raise ValueError("requested channel/split is absent from index")
        result = set()
        for channel in channels:
            result.update(self.metadata.get("elements", self.metadata["checkpoint"]["elements"])[channel][split])
        return tuple(sorted(int(z) for z in result))

    def record(self, channel, split, ordinal):
        self._check_process()
        _integer(ordinal, "record ordinal")
        if ordinal >= self.counts(channel, split):
            raise IndexError(ordinal)
        row = self.db.execute("""SELECT r.*,f.path,f.size file_size,f.mtime_ns file_mtime_ns
          FROM records r JOIN files f ON f.id=r.file_id WHERE r.channel=? AND r.split=? AND r.ordinal=?""",
                              (channel, split, ordinal)).fetchone()
        if row is None:
            raise ValueError("index ordinal is missing")
        return dict(row)

    def iter_records(self, channels, split):
        self._check_process()
        channels = _channels(channels)
        if not set(channels).issubset(self.channels) or split not in SPLITS:
            raise ValueError("requested channel/split is absent from index")
        for channel in channels:
            cursor = self.db.execute("""SELECT r.*,f.path,f.size file_size,f.mtime_ns file_mtime_ns
              FROM records r JOIN files f ON f.id=r.file_id WHERE r.channel=? AND r.split=? ORDER BY r.ordinal""",
                                     (channel, split))
            try:
                for row in cursor:
                    yield dict(row)
            finally:
                cursor.close()

    def verify_sources(self):
        self._check_process()
        _verify_files(self.db)
        config = self.metadata["config"]
        if _stat(config["manifest"]) != config["manifest_stat"]:
            raise ValueError("indexed manifest file changed")

    def read_atoms(self, record, verify=True):
        self._check_process()
        key = record["path"]
        handle = self._handles.pop(key, None)
        if handle is None:
            if _stat(key) != {"size": record["file_size"], "mtime_ns": record["file_mtime_ns"]}:
                raise ValueError("indexed export file changed: " + key)
            handle = Path(key).open("rb", buffering=1024 * 1024)
        self._handles[key] = handle
        if len(self._handles) > 16:
            self._handles.popitem(last=False)[1].close()
        handle.seek(record["offset"])
        raw = handle.read(record["length"])
        if len(raw) != record["length"] or (verify and hashlib.sha256(raw).hexdigest() != record["sha256"]):
            raise ValueError("indexed frame SHA256 mismatch at record %s" % record["id"])
        return _parse_frame(raw, record["natoms"])

    def close(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        if getattr(self, "db", None) is not None:
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
