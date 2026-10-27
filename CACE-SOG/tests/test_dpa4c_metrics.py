"""Paired metrics use all samples and never silently mix source frames."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from dpa4c_sog import metrics


def make_row(key, n, reference_energy, dpa_energy, reference_force):
    f = np.asarray(reference_force, dtype=float).reshape(n, 3)
    return {"frame_key": key, "split": "test", "frame_sha256": "a" * 64, "natoms": n,
            "reference_energy_eV": reference_energy, "dpa_energy_eV": dpa_energy,
            "sog_energy_eV": reference_energy,
            "reference_forces_eV_A": f.tolist(), "dpa_forces_eV_A": (f + 1).tolist(),
            "sog_forces_eV_A": f.tolist()}


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_real_plots_and_correct_reference_weighting(tmp_path):
    path = tmp_path / "predictions.jsonl"
    write_rows(path, [make_row("one", 2, -4, -3, range(6)),
                      make_row("two", 1, -1, -1.5, range(6, 9))])
    report = metrics.evaluate_predictions(path, tmp_path / "result", bins=32)
    assert report["status"] == "complete"
    assert report["paired_frames"] == 2 and report["atoms"] == 3
    energy = report["metrics"]["dpa"]["energy"]
    assert energy["count"] == 2 and energy["mean_difference"] == pytest.approx(0)
    assert energy["mae"] == pytest.approx(0.5)
    assert energy["rmse"] == pytest.approx(0.5) and energy["r2"] == pytest.approx(0)
    force = report["metrics"]["dpa"]["force"]
    assert force["count"] == 9 and force["mean_difference"] == pytest.approx(1)
    assert force["mae"] == pytest.approx(1) and force["rmse"] == pytest.approx(1)
    assert force["r2"] == pytest.approx(0.85)
    assert report["metrics"]["sog"]["force"]["rmse"] == pytest.approx(0)
    assert (tmp_path / "result/dpa_parity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (tmp_path / "result/sog_parity.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.loads((tmp_path / "result/metrics.json").read_text())["input_sha256"] == report["input_sha256"]


def test_zero_variance_has_undefined_r2(tmp_path, monkeypatch):
    path = tmp_path / "single.jsonl"
    write_rows(path, [make_row("one", 1, -1, -2, [0, 0, 0])])
    monkeypatch.setattr(metrics, "_render", lambda *args: None)
    result = metrics.evaluate_predictions(path, tmp_path / "result", bins=16)
    assert result["metrics"]["dpa"]["energy"]["r2"] is None
    assert result["metrics"]["dpa"]["force"]["r2"] is None


@pytest.mark.parametrize("defect", ["duplicate", "wrong_split", "nan", "wrong_shape", "bad_sha", "different_sha", "bool_force"])
def test_invalid_or_duplicate_predictions_never_publish_output(tmp_path, defect):
    first = make_row("one", 1, -1, -2, [0, 1, 2])
    second = make_row("two", 1, -2, -3, [1, 2, 3])
    if defect == "duplicate":
        second["frame_key"] = first["frame_key"]
    if defect == "wrong_split":
        second["split"] = "train"
    if defect == "nan":
        second["dpa_energy_eV"] = float("nan")
    if defect == "wrong_shape":
        second["sog_forces_eV_A"] = [[1, 2]]
    if defect == "bad_sha":
        second["sha256"] = "not-a-digest"
    if defect == "different_sha":
        second["source_frame_sha256"] = "b" * 64
    if defect == "bool_force":
        second["dpa_forces_eV_A"] = [[False, 2, 3]]
    source = tmp_path / "bad.jsonl"
    write_rows(source, [first, second])
    with pytest.raises(ValueError):
        metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    assert not (tmp_path / "result").exists()


def test_plot_density_contains_every_frame_and_force_component(tmp_path, monkeypatch):
    source = tmp_path / "many.jsonl"
    write_rows(source, (make_row(str(i), 1, -1 - i / 100, -2 - i / 100, [i, -i, 0]) for i in range(1000)))
    counts = []
    def capture(path, model, frames, bounds, histograms, result):
        counts.append((model, frames, int(histograms[model]["energy"].sum()),
                       int(histograms[model]["force"].sum())))
    monkeypatch.setattr(metrics, "_render", capture)
    result = metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    assert counts == [("dpa", 1000, 1000, 3000), ("sog", 1000, 1000, 3000)]
    assert result["paired_frames"] == 1000


def test_mutated_input_between_passes_does_not_publish(tmp_path, monkeypatch):
    source = tmp_path / "paired.jsonl"
    write_rows(source, [make_row("one", 1, -1, -2, [0, 1, 2])])
    original = metrics._lines
    calls = 0
    def changed(path, digest):
        nonlocal calls
        calls += 1
        if calls == 2:
            write_rows(source, [make_row("two", 1, -1, -2, [0, 1, 2])])
        yield from original(path, digest)
    monkeypatch.setattr(metrics, "_lines", changed)
    with pytest.raises(ValueError, match="changed"):
        metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    assert not (tmp_path / "result").exists()


def test_existing_output_and_empty_input_fail_closed(tmp_path):
    source = tmp_path / "empty.jsonl"
    source.write_text("")
    with pytest.raises(ValueError, match="no paired"):
        metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError, match="exists"):
        metrics.evaluate_predictions(source, tmp_path / "existing", bins=16)


def test_cli_creates_report_and_plots(tmp_path):
    source = tmp_path / "pair.jsonl"
    write_rows(source, [make_row("one", 1, -1, -2, [0, 1, 2]),
                        make_row("two", 1, -2, -2, [1, 2, 3])])
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8="1")
    run = subprocess.run([sys.executable, "-m", "dpa4c_sog.metrics", "--input", str(source),
                          "--output", str(tmp_path / "result"), "--bins", "16"],
                         cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["paired_frames"] == 2
    assert (tmp_path / "result/dpa_parity.png").is_file()
    assert (tmp_path / "result/sog_parity.png").is_file()


def status_for(path, count=1):
    return {"status": "complete", "channel": "sog_candidates", "split": "test",
            "expected_frames": count, "frames_written": count,
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "index_identity": "1" * 64,
            "checkpoints": {"dpa": {"sha256": "2" * 64}, "sog": {"sha256": "3" * 64}},
            "condition_provenance": "explicit [0,1] model input assumption"}


def write_status(path, status):
    path.with_name(path.name + ".status.json").write_text(json.dumps(status), encoding="utf-8")


def test_complete_prediction_sidecar_is_verified_and_preserved(tmp_path, monkeypatch):
    source = tmp_path / "pair.jsonl"
    write_rows(source, [make_row("one", 1, -1, -2, [0, 1, 2])])
    status = status_for(source)
    write_status(source, status)
    monkeypatch.setattr(metrics, "_render", lambda *args: None)
    result = metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    provenance = result["prediction_provenance"]
    assert provenance["prediction_sha256_and_frame_counts_verified"]
    assert provenance["status"]["index_identity"] == "1" * 64
    assert provenance["status"]["checkpoints"]["dpa"]["sha256"] == "2" * 64
    assert provenance["status"]["condition_provenance"] == status["condition_provenance"]


@pytest.mark.parametrize("defect", ["partial", "wrong_hash", "wrong_count", "incomplete_count", "wrong_split", "bad_json"])
def test_bad_prediction_sidecar_cannot_produce_report(tmp_path, defect):
    source = tmp_path / "pair.jsonl"
    write_rows(source, [make_row("one", 1, -1, -2, [0, 1, 2])])
    status = status_for(source)
    if defect == "partial":
        status["status"] = "failed"
    elif defect == "wrong_hash":
        status["predictions_sha256"] = "0" * 64
    elif defect == "wrong_count":
        status["expected_frames"] = status["frames_written"] = 2
    elif defect == "incomplete_count":
        status["expected_frames"] = 2
    elif defect == "wrong_split":
        status["split"] = "val"
    write_status(source, status)
    if defect == "bad_json":
        source.with_name(source.name + ".status.json").write_text("{invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar"):
        metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    assert not (tmp_path / "result").exists()


def test_status_created_during_plot_pass_cannot_be_ignored(tmp_path, monkeypatch):
    source = tmp_path / "pair.jsonl"
    write_rows(source, [make_row("one", 1, -1, -2, [0, 1, 2])])
    original = metrics._lines
    calls = 0
    def changed(path, digest):
        nonlocal calls
        calls += 1
        if calls == 2:
            write_status(source, status_for(source))
        yield from original(path, digest)
    monkeypatch.setattr(metrics, "_lines", changed)
    monkeypatch.setattr(metrics, "_render", lambda *args: None)
    with pytest.raises(ValueError, match="sidecar changed"):
        metrics.evaluate_predictions(source, tmp_path / "result", bins=16)
    assert not (tmp_path / "result").exists()
