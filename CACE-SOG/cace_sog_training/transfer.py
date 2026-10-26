"""Seal and relocate a completed CPU-built index without reparsing all frames.

A bundle contains an SQLite snapshot and content digests, not the dataset.
Binding verifies the snapshot and every referenced data file before publishing
a NEW location-bound index. Existing training readers and per-frame hashes are
unchanged. Relocate before starting training: checkpoint identities remain bound
to the resulting index metadata.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import sqlite3
import sys
import tempfile
import time

from .data import (FORMAT_VERSION, IndexReader, _connect, _export_status, _metadata,
                   _put, _read_json, _stat)

BUNDLE_FORMAT = "elementa-index-bundle-v1"
MAX_FILES = 4096
MAX_DESCRIPTOR_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 8 * 1024 * 1024


def _emit(progress, event):
    if callable(progress):
        progress(event)
    elif progress:
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), file=sys.stderr, flush=True)


def _hash_file(path, progress, label):
    before = _stat(path)
    digest = hashlib.sha256()
    consumed = 0
    last = time.monotonic()
    _emit(progress, {"stage": "hash", "file": label, "bytes_total": before["size"]})
    with Path(path).open("rb", buffering=CHUNK_BYTES) as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            consumed += len(chunk)
            if time.monotonic() - last >= 30:
                if _stat(path) != before:
                    raise ValueError("file changed during SHA256 verification: " + str(path))
                _emit(progress, {"stage": "hash", "file": label,
                                 "bytes": consumed, "bytes_total": before["size"]})
                last = time.monotonic()
    if consumed != before["size"] or _stat(path) != before:
        raise ValueError("file changed during SHA256 verification: " + str(path))
    return {"size": consumed, "sha256": digest.hexdigest()}


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid bundle SHA256 for " + name)
    return value


def _relative(value):
    """Reject absolute, parent, empty, drive and alternate-separator paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("bundle relative_path must use nonempty forward-slash paths")
    windows = PureWindowsPath(value)
    parts = value.split("/")
    if windows.drive or windows.root or value.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError("unsafe bundle relative_path: " + value)
    if ":" in value or "\x00" in value:
        raise ValueError("unsafe bundle relative_path: " + value)
    return PurePosixPath(*parts).as_posix()


def _inside(root, relative):
    relative = _relative(relative)
    path = root.joinpath(*relative.split("/")).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("bundle path escapes its root through a symlink: " + relative) from exc
    if not path.is_file():
        raise ValueError("bundle path is not a regular file: " + relative)
    return path


def _relative_source(path, root):
    """Portable mapping; PureWindowsPath also supports unit checks on Unix."""
    if PureWindowsPath(str(root)).drive:
        source, base = PureWindowsPath(str(path)), PureWindowsPath(str(root))
    else:
        source, base = PurePosixPath(str(path)), PurePosixPath(str(root))
    try:
        value = source.relative_to(base).as_posix()
    except ValueError as exc:
        raise ValueError("indexed source is outside its export root: " + str(path)) from exc
    return _relative(value)


def _atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("x", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temp), str(path))
    finally:
        if temp.exists():
            temp.unlink()


def _quick_check(db):
    for row in db.execute("PRAGMA quick_check"):
        if row[0] != "ok":
            raise ValueError("SQLite snapshot failed integrity check: " + str(row[0]))


def _backup(source, destination, progress):
    input_db = _connect(source, readonly=True)
    output_db = sqlite3.connect(str(destination))
    last = [time.monotonic()]
    try:
        def report(status, remaining, total):
            if remaining == 0 or time.monotonic() - last[0] >= 30:
                _emit(progress, {"stage": "snapshot", "pages_copied": total - remaining, "pages_total": total})
                last[0] = time.monotonic()
        input_db.backup(output_db, pages=1024, progress=report)
        output_db.commit()
    finally:
        output_db.close()
        input_db.close()


def seal_index(database, bundle, progress=True):
    """Seal a complete index into a NEW metadata-only bundle directory.

    Builds a consistent SQLite snapshot, prepares train-only E0 in that snapshot
    and hashes all referenced files. Source index and extxyz files are read-only.
    bundle.json is published last and is the only completed-bundle marker.
    """
    from .baseline import fit_baseline

    database = Path(database).expanduser().resolve(strict=True)
    bundle = Path(bundle).expanduser().resolve()
    if bundle.exists():
        raise ValueError("bundle directory already exists; choose a NEW directory")
    with IndexReader(database) as source:
        if source.metadata["config"].get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported index format")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.mkdir()
    snapshot = bundle / "index.sqlite3"
    try:
        _backup(database, snapshot, progress)
        with IndexReader(snapshot) as reader:
            _quick_check(reader.db)
            metadata = reader.metadata
            config = metadata["config"]
            channels = tuple(config["channels"])
            count = reader.db.execute("SELECT count(*) FROM files").fetchone()[0]
            if count > MAX_FILES:
                raise ValueError("bundle file-count parser guard exceeded; no records were dropped")
            rows = [dict(row) for row in reader.db.execute("SELECT * FROM files ORDER BY id")]
            _emit(progress, {"stage": "train_baseline", "channels": list(channels)})
            e0, fit = fit_baseline(reader, channels)
            reader.verify_sources()
        prepared = {"channels": list(channels), "e0": {str(z): value for z, value in e0.items()}, "fit": fit}
        db = _connect(snapshot)
        try:
            _put(db, "prepared_baseline", prepared)
            db.commit()
        finally:
            db.close()

        root = Path(config["export_dir"]).resolve(strict=True)
        manifest = _inside(root, "manifest.jsonl")
        summary = _inside(root, "summary.json")
        manifest_hash = _hash_file(manifest, progress, "manifest.jsonl")
        summary_hash = _hash_file(summary, progress, "summary.json")
        if manifest_hash["sha256"] != config["manifest_sha256"] or summary_hash["sha256"] != config["summary_sha256"]:
            raise ValueError("manifest/summary changed since the index was built")
        expected, _ = _export_status(root, channels)
        if expected != metadata["counts"]:
            raise ValueError("export summary counts differ from the complete index")
        descriptors = []
        relative_seen, resolved_seen = set(), set()
        for row in rows:
            relative = _relative_source(row["path"], config["export_dir"])
            folded = relative.casefold()
            if folded in relative_seen or relative in ("manifest.jsonl", "summary.json", "run_status.json"):
                raise ValueError("colliding bundle data relative_path: " + relative)
            relative_seen.add(folded)
            path = _inside(root, relative)
            if str(path) in resolved_seen:
                raise ValueError("multiple bundle files alias the same source through symlinks")
            resolved_seen.add(str(path))
            if _stat(path) != {"size": row["size"], "mtime_ns": row["mtime_ns"]}:
                raise ValueError("indexed export file changed: " + str(path))
            descriptor = {"file_id": int(row["id"]), "relative_path": relative}
            descriptor.update(_hash_file(path, progress, relative))
            descriptors.append(descriptor)
        status_hash = None
        if (root / "run_status.json").exists():
            status_hash = _hash_file(_inside(root, "run_status.json"), progress, "run_status.json")
        # Recheck source bindings after all I/O before declaring completion.
        with IndexReader(snapshot) as reader:
            reader.verify_sources()
        payload = {"format": BUNDLE_FORMAT, "status": "complete", "index_format_version": FORMAT_VERSION,
                   "source_export_dir": config["export_dir"], "channels": list(channels),
                   "counts": metadata["counts"], "indexed_frames": metadata["summary"]["indexed_frames"],
                   "snapshot": {"relative_path": "index.sqlite3", **_hash_file(snapshot, progress, "index.sqlite3")},
                   "manifest": {"relative_path": "manifest.jsonl", **manifest_hash},
                   "summary": {"relative_path": "summary.json", **summary_hash},
                   "files": descriptors, "dataset_files_copied": False,
                   "verification": "full-file SHA256; existing per-frame SHA256 remains enabled",
                   "prepared_baseline": {"channels": list(channels), "fit": fit}}
        if status_hash is not None:
            payload["run_status"] = {"relative_path": "run_status.json", **status_hash}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf8")
        if len(encoded) > MAX_DESCRIPTOR_BYTES:
            raise ValueError("bundle descriptor parser guard exceeded")
        _atomic_json(bundle / "bundle.json", payload)
        _emit(progress, {"stage": "sealed", "bundle": str(bundle), "files": len(descriptors)})
        return {"status": "complete", "bundle": str(bundle), "indexed_frames": payload["indexed_frames"],
                "channels": list(channels), "data_files": len(descriptors), "dataset_files_copied": False,
                "snapshot_sha256": payload["snapshot"]["sha256"]}
    except BaseException as exc:
        try:
            _atomic_json(bundle / "seal_status.json", {"status": "failed", "error": str(exc)[:4096]})
        except OSError:
            pass  # Preserve the primary error if the output disk is unavailable.
        raise


def _load_bundle(root):
    path = _inside(root, "bundle.json")
    before = _stat(path)
    if before["size"] > MAX_DESCRIPTOR_BYTES:
        raise ValueError("bundle descriptor exceeds bounded parser limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_DESCRIPTOR_BYTES + 1)
    if len(raw) > MAX_DESCRIPTOR_BYTES:
        raise ValueError("bundle descriptor exceeds bounded parser limit")
    if _stat(path) != before:
        raise ValueError("bundle descriptor changed while reading")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict) or value.get("format") != BUNDLE_FORMAT or value.get("status") != "complete":
        raise ValueError("not a complete ELEMENTA index bundle")
    if value.get("index_format_version") != FORMAT_VERSION:
        raise ValueError("unsupported bundled index format")
    files = value.get("files")
    if not isinstance(files, list) or len(files) > MAX_FILES:
        raise ValueError("invalid or oversized bundle file list")
    return value, hashlib.sha256(raw).hexdigest()


def _verify_descriptor(root, descriptor, progress):
    if not isinstance(descriptor, dict):
        raise ValueError("invalid bundle file descriptor")
    relative = _relative(descriptor.get("relative_path"))
    expected = _sha(descriptor.get("sha256"), relative)
    size = descriptor.get("size")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("invalid bundled file size")
    path = _inside(root, relative)
    if _stat(path)["size"] != size:
        raise ValueError("bundled file size mismatch: " + relative)
    actual = _hash_file(path, progress, relative)
    if actual["sha256"] != expected:
        raise ValueError("bundled file SHA256 mismatch: " + relative)
    return path, _stat(path)


def _publish_new(temp, destination):
    # Same-directory hard-link publication is atomic and fails if destination
    # already exists. os.replace would silently overwrite an existing index.
    try:
        os.link(str(temp), str(destination))
    except FileExistsError as exc:
        raise ValueError("target index already exists; no overwrite is permitted") from exc
    except OSError as exc:
        raise OSError("target filesystem must support same-directory hard links for atomic no-overwrite publication") from exc
    temp.unlink()


def bind_index(bundle, export_dir, database, progress=True):
    """Verify binary-transferred files and publish a NEW bound schema-1 index.

    Performs full sequential hashes, not all-frame ASE parsing/graph creation.
    No target DB is published on failed validation. Source bundle is unchanged.
    """
    bundle = Path(bundle).expanduser().resolve(strict=True)
    root = Path(export_dir).expanduser().resolve(strict=True)
    database = Path(database).expanduser().resolve()
    if database.exists():
        raise ValueError("target index already exists; choose a NEW database")
    try:
        database.relative_to(bundle)
    except ValueError:
        pass
    else:
        raise ValueError("target database cannot be written inside the sealed source bundle")
    payload, descriptor_sha = _load_bundle(bundle)
    snapshot, snapshot_stat = _verify_descriptor(bundle, payload.get("snapshot"), progress)
    inspection = _connect(snapshot, readonly=True)
    try:
        metadata = _metadata(inspection)
        if metadata.get("status") != "complete" or metadata.get("config", {}).get("format_version") != FORMAT_VERSION:
            raise ValueError("bundle snapshot is not a complete compatible index")
        if inspection.execute("SELECT count(*) FROM files").fetchone()[0] > MAX_FILES:
            raise ValueError("snapshot file-count parser guard exceeded")
        rows = {int(row["id"]): dict(row) for row in inspection.execute("SELECT * FROM files")}
        if len(rows) > MAX_FILES:
            raise ValueError("snapshot file-count parser guard exceeded")
    finally:
        inspection.close()
    if payload.get("channels") != metadata["config"].get("channels") or payload.get("counts") != metadata.get("counts"):
        raise ValueError("bundle descriptor and snapshot scope/counts differ")
    if payload.get("indexed_frames") != metadata["summary"]["indexed_frames"]:
        raise ValueError("bundle descriptor frame count differs from snapshot")
    for field, filename, key in (("manifest", "manifest.jsonl", "manifest_sha256"),
                                 ("summary", "summary.json", "summary_sha256")):
        descriptor = payload.get(field)
        if not isinstance(descriptor, dict) or descriptor.get("relative_path") != filename:
            raise ValueError("bundle is missing its complete " + filename)
        if descriptor.get("sha256") != metadata["config"].get(key):
            raise ValueError("bundle descriptor and indexed " + filename + " SHA256 differ")
    manifest, manifest_stat = _verify_descriptor(root, payload["manifest"], progress)
    summary, summary_stat = _verify_descriptor(root, payload["summary"], progress)
    checked = [(manifest, manifest_stat), (summary, summary_stat), (snapshot, snapshot_stat)]
    if "run_status" in payload:
        if payload["run_status"].get("relative_path") != "run_status.json":
            raise ValueError("invalid run_status descriptor")
        status, status_stat = _verify_descriptor(root, payload["run_status"], progress)
        checked.append((status, status_stat))
    expected, _ = _export_status(root, tuple(payload["channels"]))
    if expected != metadata["counts"]:
        raise ValueError("target export summary counts differ from index")

    assignments = {}
    relative_seen, resolved_seen = set(), set()
    for descriptor in payload["files"]:
        if not isinstance(descriptor, dict):
            raise ValueError("invalid data file descriptor")
        file_id = descriptor.get("file_id")
        if not isinstance(file_id, int) or isinstance(file_id, bool) or file_id not in rows or file_id in assignments:
            raise ValueError("invalid or duplicate bundled file_id")
        relative = _relative(descriptor.get("relative_path"))
        source_relative = _relative_source(rows[file_id]["path"], metadata["config"]["export_dir"])
        if relative != source_relative or descriptor.get("size") != rows[file_id]["size"]:
            raise ValueError("bundle data file mapping differs from snapshot")
        folded = relative.casefold()
        if folded in relative_seen or relative in ("manifest.jsonl", "summary.json", "run_status.json"):
            raise ValueError("colliding bundle relative_path")
        relative_seen.add(folded)
        path, current = _verify_descriptor(root, descriptor, progress)
        if str(path) in resolved_seen:
            raise ValueError("multiple target data files alias the same file through symlinks")
        resolved_seen.add(str(path))
        checked.append((path, current))
        assignments[file_id] = (path, current)
    if set(assignments) != set(rows):
        raise ValueError("bundle omits an indexed data file")
    for path, before in checked:
        if _stat(path) != before:
            raise ValueError("verified input changed before binding: " + str(path))

    database.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix="." + database.name + ".binding-", suffix=".sqlite3", dir=str(database.parent))
    os.close(handle)
    temporary = Path(temporary)
    try:
        shutil.copyfile(snapshot, temporary)
        if _hash_file(temporary, progress, "copied index snapshot")["sha256"] != payload["snapshot"]["sha256"]:
            raise ValueError("copied index snapshot SHA256 mismatch")
        db = _connect(temporary)
        try:
            db.execute("BEGIN IMMEDIATE")
            for file_id, (path, current) in assignments.items():
                db.execute("UPDATE files SET path=?,size=?,mtime_ns=? WHERE id=?",
                           (str(path), current["size"], current["mtime_ns"], file_id))
            config = dict(metadata["config"])
            config.update(export_dir=str(root), manifest=str(manifest), manifest_stat=manifest_stat)
            _put(db, "config", config)
            _put(db, "export_dir", str(root))
            output_summary = dict(metadata["summary"])
            output_summary.update(database=str(database), export_dir=str(root))
            _put(db, "summary", output_summary)
            _put(db, "relocation", {"bundle_format": BUNDLE_FORMAT,
                 "bundle_descriptor_sha256": descriptor_sha, "source_snapshot_sha256": payload["snapshot"]["sha256"],
                 "source_export_dir": metadata["config"]["export_dir"],
                 "verification": "full snapshot/manifest/summary/referenced-file SHA256 before path/stat rebinding",
                 "per_frame_SHA256_preserved": True})
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()
        for path, before in checked:
            if _stat(path) != before:
                raise ValueError("verified input changed before index publication: " + str(path))
        with IndexReader(temporary) as reader:
            reader.verify_sources()
        _publish_new(temporary, database)
        result = {"status": "complete", "database": str(database), "export_dir": str(root),
                  "channels": payload["channels"], "indexed_frames": payload["indexed_frames"],
                  "data_files": len(assignments), "full_file_SHA256_verified": True,
                  "per_frame_SHA256_preserved": True, "ASE_frames_reparsed": 0,
                  "prepared_baseline": "prepared_baseline" in metadata}
        _emit(progress, {"stage": "bound", **result})
        return result
    finally:
        if temporary.exists():
            temporary.unlink()
