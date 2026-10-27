"""Bound-index frame identity stays stable while metadata location changes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from cace_sog_training.data import IndexReader, build_index
from cace_sog_training.transfer import bind_index, seal_index
from dpa4c_sog.alignment import verify_alignment
from dpa4c_sog.convert import convert
from dpa4c_sog import alignment
from elementa_fixtures import exported_fixture


@pytest.fixture
def aligned_inputs(tmp_path):
    source = tmp_path / "original Windows export"
    exported_fixture(source, channels=("sog_candidates",), train=2, val=2, test=2)
    source_db = tmp_path / "source.sqlite3"
    build_index(source, source_db, channels=["sog_candidates"])
    converted = tmp_path / "converted"
    convert(source_db, converted, (0, 1), set_frames=2)
    bundle = tmp_path / "bundle"
    seal_index(source_db, bundle, progress=False)
    remote = tmp_path / "SCNet" / "export"
    shutil.copytree(source, remote)
    bound = tmp_path / "SCNet" / "index.sqlite3"
    bind_index(bundle, remote, bound, progress=False)
    return bound, converted


def read_rows(converted):
    path = converted / "frames.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_rows(converted, rows):
    (converted / "frames.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_rebound_index_and_conversion_match_per_frame(aligned_inputs, tmp_path):
    bound, converted = aligned_inputs
    output = tmp_path / "aligned.json"
    report = verify_alignment(bound, converted, output, progress_seconds=1000)
    assert report["status"] == "complete" and report["frames_checked"] == 6
    assert report["counts"] == {"train": 2, "val": 2, "test": 2}
    assert report["metadata_identities_differ"] is True
    assert report["source_index_metadata_sha256"] != report["bound_index_identity"]
    assert report["converted_dir"] == str(converted.resolve())
    assert report["conversion_status_sha256"] == hashlib.sha256((converted / "conversion_status.json").read_bytes()).hexdigest()
    assert report["frames_manifest_sha256"] == hashlib.sha256((converted / "frames.jsonl").read_bytes()).hexdigest()
    assert report["converted_npy_contents_verified"] is False
    assert report["source_frame_bytes_rehashed"] is False
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("defect", ["missing", "duplicate", "sha", "split", "permutation", "source_id",
                                    "manifest_line", "component", "system_escape", "set_escape", "row_gap", "slot_duplicate", "natoms", "composition"])
def test_manifest_mismatch_never_publishes_report(aligned_inputs, tmp_path, defect):
    bound, converted = aligned_inputs
    rows = read_rows(converted)
    if defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "sha":
        rows[0]["sha256"] = "b" * 64
    elif defect == "split":
        rows[0]["split"] = "train" if rows[0]["split"] != "train" else "test"
    elif defect == "permutation":
        rows[0]["canonical_to_source"] = [0, 0]
    elif defect == "source_id":
        rows[0]["source_id"] += 1000
    elif defect == "manifest_line":
        rows[0]["manifest_line"] += 1000
    elif defect == "component":
        rows[0]["component_id"] = "wrong-parent"
    elif defect == "system_escape":
        rows[0]["system"] = "../other"
    elif defect == "set_escape":
        rows[0]["set"] = "../../set.000"
    elif defect == "row_gap":
        rows[0]["row"] = 5
    elif defect == "slot_duplicate":
        rows[1]["row"] = rows[0]["row"]
    elif defect == "natoms":
        rows[0]["natoms"] = 3
    elif defect == "composition":
        rows[0]["system"] = "n2_" + "0" * 24
    write_rows(converted, rows)
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        verify_alignment(bound, converted, output, progress_seconds=1000)
    assert not output.exists()


def test_bad_or_incomplete_conversion_status_is_rejected(aligned_inputs, tmp_path):
    bound, converted = aligned_inputs
    path = converted / "conversion_status.json"
    data = json.loads(path.read_text())
    data["status"] = "partial"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete"):
        verify_alignment(bound, converted, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_existing_report_is_not_overwritten(aligned_inputs, tmp_path):
    bound, converted = aligned_inputs
    output = tmp_path / "existing.json"
    output.write_text("owned")
    with pytest.raises(ValueError, match="exists"):
        verify_alignment(bound, converted, output)
    assert output.read_text() == "owned"


def test_converted_status_change_mid_scan_fails(aligned_inputs, tmp_path, monkeypatch):
    bound, converted = aligned_inputs
    source = converted / "conversion_status.json"
    original = alignment._permutation
    called = False
    def change_then_check(permutation, natoms):
        nonlocal called
        if not called:
            called = True
            source.write_text(json.dumps(json.loads(source.read_text()), indent=3))
        original(permutation, natoms)
    monkeypatch.setattr(alignment, "_permutation", change_then_check)
    with pytest.raises(ValueError, match="status changed"):
        verify_alignment(bound, converted, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_alignment_does_not_reparse_atomic_arrays(aligned_inputs, tmp_path, monkeypatch):
    bound, converted = aligned_inputs
    def fail_if_called(*args, **kwargs):
        raise AssertionError("alignment must not read or parse atom arrays")
    monkeypatch.setattr(IndexReader, "read_atoms", fail_if_called)
    assert verify_alignment(bound, converted, tmp_path / "aligned.json")["frames_checked"] == 6


def test_cli_checks_bound_index_and_writes_report(aligned_inputs, tmp_path):
    bound, converted = aligned_inputs
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8="1")
    process = subprocess.run([sys.executable, "-m", "dpa4c_sog.alignment", "--database", str(bound),
        "--converted", str(converted), "--output", str(tmp_path / "report.json")],
        cwd=root, env=env, text=True, encoding="utf-8", capture_output=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads(process.stdout)["frames_checked"] == 6
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "complete"
