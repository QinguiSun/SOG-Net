"""Offline plans, real DPDispatcher object construction, and remote retry guards."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from cace_sog_training.dispatch import _REMOTE_RUNNER, dispatch


SHA = "b249fc9bf1e92f5a0d4a1432acd26d7f4f0db89f"


def config(tmp_path, stage="train"):
    return {
        "stage": stage, "expected_git_commit": SHA, "local_work_dir": str(tmp_path / "controller"),
        "remote_project_dir": "/work/project with spaces/CACE-SOG",
        "remote_run_dir": "/work/runs/experiment", "remote_database": "/work/data/index.sqlite3",
        "remote_conda_env": "/work/envs/mlip", "remote_conda_sh": "/work/conda/etc/profile.d/conda.sh",
        "remote_bundle": "/work/data/bundle", "remote_export_dir": "/work/data/export",
        "machine": {"remote_root": "/work/dispatcher", "remote_profile": {
            "hostname": "cluster.example.org", "username": "example", "port": 65032,
            "key_filename": "C:/Users/example/.ssh/private-key-location",
        }},
        "resources": {"cpus_per_task": 4, "memory": "32G", "time_limit": "1-00:00:00"},
        "options": {"epochs": 3, "batch_size": 2} if stage == "train" else {},
    }


def test_offline_plan_does_not_import_dpdispatcher_or_connect(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "dpdispatcher", None)
    cfg = config(tmp_path)
    result = dispatch(cfg)
    assert result["status"] == "planned" and result["submitted"] is False
    assert result["forward_files"] == ["run_stage.sh", "run_stage.py", "runtime_config.json"]
    assert result["cleanup"] is False
    assert result["resources"]["cpu_per_node"] == 1
    assert result["resources"]["gpu_per_node"] == 1
    assert "#SBATCH --cpus-per-task=4" in result["resources"]["custom_flags"]
    assert not any("account" in line for line in result["resources"]["custom_flags"])
    folder = Path(result["local_task_dir"])
    assert (folder / "plan.json").is_file()
    assert b"\r\n" not in (folder / "run_stage.sh").read_bytes()
    assert 'conda activate /work/envs/mlip' in (folder / "run_stage.sh").read_text()
    for filename in ["runtime_config.json", "plan.json", "run_stage.sh", "run_stage.py"]:
        text = (folder / filename).read_text()
        assert "private-key-location" not in text
        assert "cluster.example.org" not in text
    assert "private-key-location" not in json.dumps(result)
    assert len(list(folder.iterdir())) == 4


def test_config_path_relative_work_dir_and_windows_bom(tmp_path):
    cfg = config(tmp_path, "bind")
    cfg["local_work_dir"] = "local work"
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(cfg), encoding="utf-8-sig")
    plan = dispatch(path)
    assert Path(plan["local_task_dir"]).parent == tmp_path / "local work"


@pytest.mark.parametrize("field,value", [
    ("remote_run_dir", "/work/run/../../etc"),
    ("remote_project_dir", "C:\\project"),
    ("remote_conda_env", "/work/env\nmalicious"),
    ("remote_database", "/"),
    ("expected_git_commit", "main"),
    ("task_name", "../escape"),
    ("options", {"run_dir": "/overwrite"}),
    ("options", {"epochs": True}),
    ("options", {"n_dl": float("nan")}),
    ("download_checkpoints", "false"),
])
def test_invalid_paths_and_options_rejected(tmp_path, field, value):
    cfg = config(tmp_path)
    cfg[field] = value
    with pytest.raises(ValueError):
        dispatch(cfg)


def test_credentials_and_raw_scheduler_flags_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg["machine"]["remote_profile"]["password"] = "do-not-print-this"
    with pytest.raises(ValueError) as caught:
        dispatch(cfg)
    assert "do-not-print-this" not in str(caught.value)
    cfg = config(tmp_path)
    cfg["resources"]["time_limit"] = "1:00:00\n#SBATCH --account=other"
    with pytest.raises(ValueError):
        dispatch(cfg)


def test_spin_review_requires_explicit_policy(tmp_path):
    cfg = config(tmp_path)
    cfg["channels"] = ["spin_review"]
    with pytest.raises(ValueError, match="consistent-state"):
        dispatch(cfg)
    cfg["options"].update(spin_policy="consistent-state", spin_policy_note="reviewed single electronic state")
    assert dispatch(cfg)["status"] == "planned"


def test_submit_constructs_actual_dpdispatcher_api_without_ssh(tmp_path, monkeypatch):
    # DPDispatcher creates its recovery directory on import. Keep it in this test.
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "test-home")
    dp = pytest.importorskip("dpdispatcher")
    cfg = config(tmp_path)
    local = dp.Machine.load_from_dict({"batch_type": "Shell", "context_type": "LazyLocalContext",
                                      "local_root": cfg["local_work_dir"]})
    captured = {}

    def load_machine(settings):
        captured["machine"] = settings
        return local

    def run(self, **kwargs):
        captured["submission"] = self
        captured["kwargs"] = kwargs
        return {"secret": "must-not-return-dpdispatcher-state"}

    monkeypatch.setattr(dp.Machine, "load_from_dict", staticmethod(load_machine))
    monkeypatch.setattr(dp.Submission, "run_submission", run)
    plan = dispatch(cfg, submit=True)
    assert captured["machine"]["context_type"] == "SSHContext"
    assert captured["machine"]["batch_type"] == "Slurm"
    submission = captured["submission"]
    assert submission.forward_common_files == []
    assert submission.belonging_tasks[0].forward_files == plan["forward_files"]
    assert submission.belonging_tasks[0].command == "bash run_stage.sh"
    assert captured["kwargs"] == {"clean": False, "check_interval": 60}
    assert plan["submitted"] is True
    assert "must-not-return" not in json.dumps(plan)


def test_submit_failure_attempts_download_and_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "test-home")
    dp = pytest.importorskip("dpdispatcher")
    cfg = config(tmp_path)
    local = dp.Machine.load_from_dict({"batch_type": "Shell", "context_type": "LazyLocalContext",
                                      "local_root": cfg["local_work_dir"]})
    monkeypatch.setattr(dp.Machine, "load_from_dict", staticmethod(lambda settings: local))
    downloaded = []
    def failed(self, **kwargs):
        raise RuntimeError("simulated scheduler failure")
    monkeypatch.setattr(dp.Submission, "run_submission", failed)
    monkeypatch.setattr(dp.Submission, "download_jobs", lambda self, **kwargs: downloaded.append(kwargs))
    with pytest.raises(RuntimeError, match="simulated scheduler failure"):
        dispatch(cfg, submit=True)
    assert downloaded == [{"include_failed": True}]
    plan = json.loads(next((tmp_path / "controller").glob("*/plan.json")).read_text())
    assert plan["status"] == "failed" and plan["cleanup"] is False


def runner(tmp_path, monkeypatch, stage="train", returncode=0):
    path = tmp_path / "run_stage.py"
    path.write_text(_REMOTE_RUNNER)
    spec = importlib.util.spec_from_file_location("remote_dispatch_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    remote = tmp_path / "persistent remote"
    database = tmp_path / "index.sqlite3"
    if stage != "bind":
        database.write_text("not read by mocked command")
    payload = {
        "stage": stage, "expected_git_commit": SHA, "remote_project_dir": str(tmp_path),
        "remote_run_dir": str(remote), "remote_database": str(database),
        "remote_bundle": str(tmp_path / "bundle"), "remote_export_dir": str(tmp_path / "export"),
        "channels": ["sog_candidates"], "options": {}, "resume_policy": "auto",
        "download_checkpoints": False, "task_name": "test-task",
    }
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kwargs: SHA + "\n" if "rev-parse" in args else "")
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append(command)
            if stage == "train":
                train_dir = remote / "train"
                train_dir.mkdir(parents=True, exist_ok=True)
                (train_dir / "run_status.json").write_text('{"status":"failed"}')
        def wait(self):
            return returncode

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    config_path = tmp_path / "runtime_config.json"
    config_path.write_text(json.dumps(payload))
    return module, payload, config_path, calls


def test_remote_failure_keeps_exit_code_and_collects_metadata(tmp_path, monkeypatch):
    module, payload, path, calls = runner(tmp_path, monkeypatch, returncode=7)
    assert module.run(path) == 7
    assert (tmp_path / "results/run_status.json").is_file()
    result = json.loads((tmp_path / "results/dispatch_status.json").read_text())
    assert result["status"] == "failed" and result["returncode"] == 7
    assert "--run-dir" in calls[0]
    assert calls[0][calls[0].index("--run-dir") + 1] == str(Path(payload["remote_run_dir"]) / "train")
    assert module.run(path) == 1
    assert len(calls) == 1  # Retry cannot restart a failed attempt without last.pt.


def test_remote_auto_resumes_persistent_checkpoint_and_download_is_opt_in(tmp_path, monkeypatch):
    module, payload, path, calls = runner(tmp_path, monkeypatch)
    train = Path(payload["remote_run_dir"]) / "train"
    train.mkdir(parents=True)
    checkpoint = train / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert module.run(path) == 0
    assert calls[0][calls[0].index("--resume") + 1] == str(checkpoint)
    assert not (tmp_path / "results/last.pt").exists()
    payload["download_checkpoints"] = True
    path.write_text(json.dumps(payload))
    assert module.run(path) == 0
    assert (tmp_path / "results/last.pt").read_bytes() == b"checkpoint"


def test_remote_git_mismatch_never_executes_stage(tmp_path, monkeypatch):
    module, payload, path, calls = runner(tmp_path, monkeypatch)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "0" * 40)
    assert module.run(path) == 1
    assert calls == []


def test_bind_does_not_retry_or_overwrite_existing_database(tmp_path, monkeypatch):
    module, payload, path, calls = runner(tmp_path, monkeypatch, stage="bind")
    assert module.run(path) == 0
    assert calls[0][3] == "bind"
    assert module.run(path) == 1
    assert len(calls) == 1
    Path(payload["remote_database"]).write_text("existing database")
    marker = Path(payload["remote_run_dir"]) / ".dispatch/test-task.json"
    marker.unlink()
    assert module.run(path) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("stage,output_name", [("smoke", "smoke"), ("converge", "converge"), ("evaluate", "test")])
def test_remote_stage_command_and_existing_checkpoint(tmp_path, monkeypatch, stage, output_name):
    module, payload, path, calls = runner(tmp_path, monkeypatch, stage=stage)
    assert module.run(path) == 0
    assert calls[0][3] == stage
    assert calls[0][calls[0].index("--output") + 1] == str(Path(payload["remote_run_dir"]) / output_name)
    if stage == "evaluate":
        assert Path(calls[0][calls[0].index("--checkpoint") + 1]).parts[-2:] == ("train", "best.pt")
