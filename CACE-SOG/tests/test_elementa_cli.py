"""Run the user-facing terminal workflow on tiny real periodic structures."""
import json
import os
from pathlib import Path
import subprocess
import sys

from elementa_fixtures import exported_fixture


ROOT = Path(__file__).resolve().parents[1]


def command(*args, success=True):
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-m", "cace_sog_training", *map(str, args)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=120)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


def test_terminal_index_smoke_train_resume_evaluate_and_trained_grid(tmp_path):
    exported_fixture(tmp_path / "export")
    db = tmp_path / "index.sqlite3"
    command("index", "--export-dir", tmp_path / "export", "--database", db, "--limit-records", "3")
    failed = command("smoke", "--database", db, "--output", tmp_path / "not-run", "--device", "cpu", success=False)
    assert "partial" in failed.stderr
    command("index", "--export-dir", tmp_path / "export", "--database", db, "--resume")
    common = ("--database", db, "--device", "cpu", "--atom-basis", "2", "--radial-basis", "2",
              "--max-l", "1", "--max-nu", "2", "--n-dl", "2", "--cutoff", "3", "--dtype", "float64")
    command("environment", "--output", tmp_path / "environment.json")
    command("smoke", *common, "--output", tmp_path / "smoke", "--steps", "3", "--batch-size", "2")
    smoke = json.loads((tmp_path / "smoke/smoke.json").read_text())
    assert smoke["optimizer_steps"] == 3 and len(smoke["memory_trace_last_32_steps"]) == 3
    run = tmp_path / "run"
    train = ("train", *common, "--run-dir", run, "--epochs", "1", "--batch-size", "4", "--checkpoint-every", "1")
    command(*train, "--max-steps", "2")
    assert json.loads((run / "run_status.json").read_text())["status"] == "partial"
    command(*train, "--resume", run / "last.pt")
    assert json.loads((run / "run_status.json").read_text())["status"] == "complete"
    command("evaluate", "--database", db, "--checkpoint", run / "best.pt", "--device", "cpu",
            "--output", tmp_path / "test", "--split", "test")
    result = json.loads((tmp_path / "test/metrics.json").read_text())
    assert result["metrics"]["all"]["frames"] == 6
    assert len(result["metrics"]) == 4
    command("converge", "--database", db, "--checkpoint", run / "best.pt", "--device", "cpu",
            "--output", tmp_path / "grids", "--n-dl", "3", "2", "--samples", "2")
    assert json.loads((tmp_path / "grids/convergence.json").read_text())["checkpoint"]

