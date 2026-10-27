"""Transfer manifests catch missing, altered and extra files and unsafe paths."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from dpa4c_sog.integrity import seal, verify


def make_dataset(base):
    dataset = base / "中文 dataset"
    (dataset / "train" / "system" / "set.000").mkdir(parents=True)
    (dataset / "test" / "empty_dir").mkdir(parents=True)
    (dataset / "conversion_status.json").write_text('{"status":"complete"}', encoding="utf-8")
    # Fixed bytes make the same-size corruption assertion portable to Windows.
    (dataset / "train" / "system" / "type.raw").write_bytes(b"0 1\n")
    (dataset / "train" / "system" / "set.000" / "coord.npy").write_bytes(b"immutable file bytes")
    (dataset / "数据.jsonl").write_text('{"id":1}\n', encoding="utf-8")
    return dataset


def write_manifest(path, lines):
    raw = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in lines)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_roundtrip_binary_unicode_and_empty_directory(tmp_path):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "dataset.jsonl"
    result = seal(source, manifest)
    assert result["status"] == "complete" and result["files"] == 4
    copied = tmp_path / "远端 copy"
    shutil.copytree(source, copied)
    result2 = verify(copied, manifest, result["manifest_sha256"])
    assert result2["status"] == "complete" and result2["files_verified"] == 4
    assert result2["directories_verified"] == 5
    assert result2["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_corruption_missing_extra_files_and_empty_directories(tmp_path):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    sha = seal(source, manifest)["manifest_sha256"]
    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    file = copy / "train" / "system" / "type.raw"
    file.write_bytes(b"0 2\n")  # equal size but different content
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify(copy, manifest, sha)
    file.write_bytes(b"0 1\n")
    file.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify(copy, manifest, sha)
    file.write_bytes(b"0 1\n")
    (copy / "extra").write_bytes(b"")
    with pytest.raises(ValueError, match="extra"):
        verify(copy, manifest, sha)
    (copy / "extra").unlink()
    (copy / "extra-directory").mkdir()
    with pytest.raises(ValueError, match="extra"):
        verify(copy, manifest, sha)


def test_duplicate_and_unsafe_paths_rejected_even_if_manifest_hash_recomputed(tmp_path):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    seal(source, manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    duplicate = next(row for row in rows if row["kind"] == "file")
    duplicate_rows = rows[:-1] + [duplicate, rows[-1]]
    sha = write_manifest(manifest, duplicate_rows)
    with pytest.raises(ValueError, match="duplicate"):
        verify(source, manifest, sha)
    rows[1]["path"] = "../escape"
    sha = write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="unsafe"):
        verify(source, manifest, sha)


def test_manifest_hash_status_and_overwrite_guards(tmp_path):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    (source / "conversion_status.json").write_text('{"status":"partial"}')
    with pytest.raises(ValueError, match="complete"):
        seal(source, manifest)
    assert not manifest.exists()
    (source / "conversion_status.json").write_text('{"status":"complete"}')
    sha = seal(source, manifest)["manifest_sha256"]
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        seal(source, manifest)
    assert manifest.read_bytes() == original
    with pytest.raises(ValueError, match="manifest SHA256"):
        verify(source, manifest, "0" * 64)
    with pytest.raises(ValueError, match="64-character"):
        verify(source, manifest, "short")
    (source / "conversion_status.json").write_text('{"status":"partial"}')
    with pytest.raises(ValueError, match="complete"):
        verify(source, manifest, sha)
    with pytest.raises(ValueError, match="outside"):
        seal(source, source / "manifest.jsonl")


def test_verified_report_is_atomic_and_refuses_existing_files(tmp_path):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    sha = seal(source, manifest)["manifest_sha256"]
    report = tmp_path / "verification.json"
    value = verify(source, manifest, sha, report)
    actual = json.loads(report.read_text(encoding="utf-8"))
    assert actual == value and actual["target_dataset"] == str(source.resolve())
    assert actual["expected_manifest_sha256"] == sha and actual["bytes_verified"] > 0
    assert len(actual["conversion_status_sha256"]) == 64
    old = report.read_bytes()
    with pytest.raises(ValueError, match="exists"):
        verify(source, manifest, sha, report)
    assert report.read_bytes() == old
    with pytest.raises(ValueError, match="outside"):
        verify(source, manifest, sha, source / "verification.json")
    assert not (source / "verification.json").exists()
    with pytest.raises(ValueError, match="manifest SHA256"):
        verify(source, manifest, "0" * 64, tmp_path / "failed-report.json")
    assert not (tmp_path / "failed-report.json").exists()


def test_symlinks_and_publish_failure_do_not_mutate_data(tmp_path, monkeypatch):
    source = make_dataset(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    link = source / "escaped"
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        seal(source, manifest)
    assert not manifest.exists()
    link.unlink()
    before = (source / "conversion_status.json").read_bytes()
    def fail_link(*args, **kwargs):
        raise OSError("injected unsupported hard link")
    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="atomic hard links"):
        seal(source, manifest)
    assert not manifest.exists()
    assert (source / "conversion_status.json").read_bytes() == before
