"""Seal or verify a completed converted dataset for cross-machine transfer.

The UTF-8 JSONL manifest lives outside the dataset and is streamed. Its SHA256
must be supplied separately to verify, so transfer mistakes cannot silently
rewrite both the inventory and files. This is integrity checking, not a digital
signature or proof of the origin of an untrusted manifest. A temporary SQLite
inventory keeps path count off the Python heap and directory walking uses a
bounded-depth scandir stack. Neither operation edits source data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import stat
import sys
import tempfile
import time

FORMAT = "dpa4c-converted-sha256-v1"
BLOCK = 8 * 1024 * 1024
MAX_LINE = 64 * 1024
MAX_STATUS = 4 * 1024 * 1024
MAX_DEPTH = 64
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _now(msg, count, last):
    current = time.monotonic()
    if current - last >= 30:
        print(f"{msg}: {count:,}", file=sys.stderr, flush=True)
        return current
    return last


def _rel(path):
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("invalid inventory path")
    parts = path.split("/")
    if (path.startswith("/") or PureWindowsPath(path).drive or PureWindowsPath(path).root
            or any(p in ("", ".", "..") for p in parts)
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        raise ValueError("unsafe or noncanonical inventory path: " + path)
    return path


def _identity(path, require_file=False):
    info = os.lstat(path)
    # Windows junctions and other reparse points may not report as symlinks.
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("symlinks are not accepted: " + str(path))
    if require_file and not stat.S_ISREG(info.st_mode):
        raise ValueError("not a regular file: " + str(path))
    # Windows file IDs can exceed SQLite's signed 64-bit integer range.
    return (int(info.st_size), int(info.st_mtime_ns), str(info.st_dev), str(info.st_ino))


def _walk(root):
    """Yield all descendants with at most MAX_DEPTH open directory handles."""
    stack = [("", os.scandir(root))]
    try:
        while stack:
            parent, cursor = stack[-1]
            try:
                entry = next(cursor)
            except StopIteration:
                cursor.close()
                stack.pop()
                continue
            relative = _rel(parent + "/" + entry.name if parent else entry.name)
            if entry.is_symlink():
                raise ValueError("symlinks are not accepted: " + relative)
            if entry.is_dir(follow_symlinks=False):
                if len(stack) >= MAX_DEPTH:
                    raise ValueError("dataset directory depth exceeds limit")
                yield relative, "dir", _identity(entry.path)
                stack.append((relative, os.scandir(entry.path)))
            elif entry.is_file(follow_symlinks=False):
                yield relative, "file", _identity(entry.path, require_file=True)
            else:
                raise ValueError("dataset contains a nonregular file: " + relative)
    finally:
        for _, cursor in stack:
            cursor.close()


def _hash_file(path, expected_identity=None, label=None):
    before = _identity(path, require_file=True)
    if expected_identity is not None and before != expected_identity:
        raise ValueError("file changed since inventory: " + str(path))
    digest = hashlib.sha256()
    length = 0
    last = time.monotonic()
    with open(path, "rb", buffering=BLOCK) as source:
        opened = os.fstat(source.fileno())
        opened_id = (int(opened.st_size), int(opened.st_mtime_ns), str(opened.st_dev), str(opened.st_ino))
        if opened_id != before:
            raise ValueError("file changed while opening: " + str(path))
        for block in iter(lambda: source.read(BLOCK), b""):
            digest.update(block)
            length += len(block)
            last = _now("Hashing " + (label or str(path)) + " bytes", length, last)
        final = os.fstat(source.fileno())
        if (int(final.st_size), int(final.st_mtime_ns), str(final.st_dev), str(final.st_ino)) != before:
            raise ValueError("file changed while hashing: " + str(path))
    if length != before[0] or _identity(path, require_file=True) != before:
        raise ValueError("file changed while hashing: " + str(path))
    return digest.hexdigest(), before


def _root(dataset, manifest):
    source = Path(dataset).expanduser()
    root_info = os.lstat(source)
    if stat.S_ISLNK(root_info.st_mode) or getattr(root_info, "st_file_attributes", 0) & 0x400:
        raise ValueError("dataset root must not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("dataset must be a directory")
    manifest = Path(manifest).expanduser().absolute()
    # Even a not-yet-created manifest must resolve outside the dataset.
    resolved = manifest.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("manifest must be outside the dataset directory")
    return root, manifest


def _status(root):
    path = root / "conversion_status.json"
    before = _identity(path, require_file=True)
    if before[0] > MAX_STATUS:
        raise ValueError("conversion status exceeds bounded limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_STATUS + 1)
    if len(raw) > MAX_STATUS or _identity(path, require_file=True) != before:
        raise ValueError("conversion status changed or exceeds bounded limit")
    status = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(status, dict) or status.get("status") != "complete":
        raise ValueError("conversion_status.json must report complete")
    return hashlib.sha256(raw).hexdigest(), before


def _new_db(parent):
    temporary = tempfile.TemporaryDirectory(prefix=".dpa4c-integrity-", dir=str(parent))
    db = sqlite3.connect(str(Path(temporary.name) / "inventory.sqlite3"))
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-16384")
    db.execute("CREATE TABLE paths(path TEXT PRIMARY KEY, kind TEXT NOT NULL, size INTEGER NOT NULL, mtime INTEGER NOT NULL, dev TEXT NOT NULL, ino TEXT NOT NULL, marked INTEGER NOT NULL DEFAULT 0)")
    return temporary, db


def _inventory(root, db):
    count, last = 0, time.monotonic()
    for relative, kind, identity in _walk(root):
        db.execute("INSERT INTO paths(path,kind,size,mtime,dev,ino) VALUES(?,?,?,?,?,?)", (relative, kind, *identity))
        count += 1
        if count % 2000 == 0:
            db.commit()
        last = _now("Inventory entries", count, last)
    db.commit()
    return count


def _recheck(root, db, count):
    """Detect added, removed or locally changed entries across the scan."""
    db.execute("UPDATE paths SET marked=0")
    seen, last = 0, time.monotonic()
    for relative, kind, ident in _walk(root):
        row = db.execute("SELECT kind,size,mtime,dev,ino FROM paths WHERE path=?", (relative,)).fetchone()
        if row is None or tuple(row) != (kind, *ident):
            raise ValueError("dataset changed during inventory: " + relative)
        db.execute("UPDATE paths SET marked=1 WHERE path=?", (relative,))
        seen += 1
        last = _now("Rechecked entries", seen, last)
    if seen != count or db.execute("SELECT 1 FROM paths WHERE marked=0 LIMIT 1").fetchone() is not None:
        raise ValueError("dataset changed during inventory: entry removed")


def _line(value):
    raw = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_LINE:
        raise ValueError("manifest record exceeds bounded line size")
    return raw


def seal(dataset, manifest):
    root, destination = _root(dataset, manifest)
    if os.path.lexists(destination):
        raise ValueError("manifest exists; choose a NEW file")
    status_sha, _ = _status(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tempdir, db = _new_db(destination.parent)
    handle, tmpname = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    tmp = Path(tmpname)
    try:
        count = _inventory(root, db)
        row = db.execute("SELECT kind FROM paths WHERE path='conversion_status.json'").fetchone()
        if row != ("file",):
            raise ValueError("dataset does not contain a regular conversion_status.json")
        digest = hashlib.sha256()
        nfiles = ndirs = 0
        with os.fdopen(handle, "wb") as stream:
            handle = None
            def write(value):
                raw = _line(value)
                stream.write(raw)
                digest.update(raw)
            write({"kind": "header", "format": FORMAT, "conversion_status_sha256": status_sha})
            entries = db.execute("SELECT path,kind,size,mtime,dev,ino FROM paths ORDER BY path")
            last = time.monotonic()
            for path, kind, size, mtime, dev, ino in entries:
                if kind == "dir":
                    write({"kind": "dir", "path": path})
                    ndirs += 1
                else:
                    sha, _ = _hash_file(root.joinpath(*path.split("/")), (size, mtime, dev, ino), path)
                    if path == "conversion_status.json" and sha != status_sha:
                        raise ValueError("conversion status changed while sealing")
                    write({"kind": "file", "path": path, "size": size, "sha256": sha})
                    nfiles += 1
                last = _now("Sealed entries", nfiles + ndirs, last)
            _recheck(root, db, count)
            write({"kind": "end", "files": nfiles, "directories": ndirs})
            stream.flush()
            os.fsync(stream.fileno())
        # Hard link is atomic and refuses to replace an existing manifest.
        try:
            os.link(str(tmp), str(destination))
        except FileExistsError as exc:
            raise ValueError("manifest exists; no overwrite is allowed") from exc
        except OSError as exc:
            raise OSError("manifest filesystem must support atomic hard links") from exc
        return {"status": "complete", "manifest": str(destination), "manifest_sha256": digest.hexdigest(),
                "files": nfiles, "directories": ndirs, "conversion_status_sha256": status_sha}
    finally:
        if handle is not None:
            os.close(handle)
        db.close()
        tempdir.cleanup()
        if tmp.exists():
            tmp.unlink()


def _write_new_report(destination, result):
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=str(destination.parent))
    tmp = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(tmp), str(destination))
        except FileExistsError as exc:
            raise ValueError("verification report exists; no overwrite is allowed") from exc
        except OSError as exc:
            raise OSError("report filesystem must support atomic hard links") from exc
    finally:
        if tmp.exists():
            tmp.unlink()


def verify(dataset, manifest, expected_manifest_sha256, report=None):
    root, source = _root(dataset, manifest)
    report_path = None
    if report is not None:
        _, report_path = _root(root, report)
        if os.path.lexists(report_path):
            raise ValueError("verification report exists; choose a NEW file")
    expected = str(expected_manifest_sha256).lower()
    if not SHA_RE.fullmatch(expected):
        raise ValueError("--expected-manifest-sha256 must be a 64-character hexadecimal SHA256")
    actual_sha, before = _hash_file(source, label="manifest")
    if actual_sha != expected:
        raise ValueError("manifest SHA256 mismatch")
    status_sha, _ = _status(root)
    tempdir, db = _new_db(source.parent)
    try:
        count = _inventory(root, db)
        digest = hashlib.sha256()
        files = dirs = lines = verified_bytes = 0
        ended = False
        last = time.monotonic()
        with source.open("rb") as stream:
            while True:
                raw = stream.readline(MAX_LINE + 1)
                if not raw:
                    break
                if len(raw) > MAX_LINE:
                    raise ValueError("manifest record exceeds bounded line size")
                digest.update(raw)
                lines += 1
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("invalid manifest record")
                kind = value.get("kind")
                if lines == 1:
                    if kind != "header" or value.get("format") != FORMAT or value.get("conversion_status_sha256") != status_sha:
                        raise ValueError("manifest header or complete conversion status mismatch")
                    continue
                if ended:
                    raise ValueError("unexpected record after manifest end")
                if kind == "end":
                    if value.get("files") != files or value.get("directories") != dirs:
                        raise ValueError("manifest entry count mismatch")
                    ended = True
                    continue
                if kind not in ("dir", "file"):
                    raise ValueError("unknown manifest record kind")
                relative = _rel(value.get("path"))
                row = db.execute("SELECT kind,size,mtime,dev,ino,marked FROM paths WHERE path=?", (relative,)).fetchone()
                if row is None or row[0] != kind:
                    raise ValueError("manifest entry missing or wrong type: " + relative)
                if row[5]:
                    raise ValueError("duplicate manifest path: " + relative)
                if kind == "file":
                    size = value.get("size")
                    sha = value.get("sha256")
                    if (not isinstance(size, int) or isinstance(size, bool) or size < 0
                            or not isinstance(sha, str) or not SHA_RE.fullmatch(sha)):
                        raise ValueError("invalid file size or SHA256 in manifest")
                    if size != row[1]:
                        raise ValueError("file size mismatch: " + relative)
                    actual, _ = _hash_file(root.joinpath(*relative.split("/")), tuple(row[1:5]), relative)
                    if actual != sha:
                        raise ValueError("file SHA256 mismatch: " + relative)
                    if relative == "conversion_status.json" and actual != status_sha:
                        raise ValueError("conversion status hash mismatch")
                    files += 1
                    verified_bytes += size
                else:
                    dirs += 1
                db.execute("UPDATE paths SET marked=1 WHERE path=?", (relative,))
                last = _now("Verified entries", files + dirs, last)
        if not ended or digest.hexdigest() != expected or _identity(source, require_file=True) != before:
            raise ValueError("manifest incomplete or changed during verification")
        if db.execute("SELECT 1 FROM paths WHERE marked=0 LIMIT 1").fetchone() is not None or files + dirs != count:
            raise ValueError("dataset contains extra entries absent from the manifest")
        _recheck(root, db, count)
        result = {"status": "complete", "target_dataset": str(root), "manifest": str(source.resolve()),
                  "expected_manifest_sha256": expected, "manifest_sha256": expected,
                  "files_verified": files, "bytes_verified": verified_bytes,
                  "directories_verified": dirs, "conversion_status_sha256": status_sha,
                  "verification_scope": "contents matched when this report was created; later changes require re-verification"}
        if report_path is not None:
            _write_new_report(report_path, result)
        return result
    finally:
        db.close()
        tempdir.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("seal", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--manifest", required=True, type=Path)
        if name == "verify":
            command.add_argument("--expected-manifest-sha256", required=True)
            command.add_argument("--report", type=Path, help="write a NEW verification report outside the dataset")
    args = parser.parse_args(argv)
    try:
        result = seal(args.dataset, args.manifest) if args.command == "seal" else verify(args.dataset, args.manifest, args.expected_manifest_sha256, args.report)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; no incomplete seal is published.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
