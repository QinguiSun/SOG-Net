"""Plan small-file DPDispatcher submissions without importing GPU dependencies.

Datasets, sealed bundles, indexes, code and Conda environments must already be
present on the remote machine.  This module deliberately does not stage them.
Calling :func:`dispatch` without ``submit=True`` never connects to a server.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex


DEFAULT_CHANNELS = ("sog_candidates", "lattice_auxiliary", "pbe_lowdim_auxiliary")
STAGES = ("bind", "smoke", "converge", "train", "evaluate")
_MODEL_OPTIONS = {
    "device", "dtype", "cutoff", "n_dl", "max_modes", "atom_basis", "radial_basis",
    "max_l", "max_nu", "message_passing", "seed", "batch_size", "spin_policy",
    "spin_policy_note",
}
_OPTIONS = {
    "bind": set(),
    "smoke": _MODEL_OPTIONS | {"steps"},
    "converge": _MODEL_OPTIONS | {"samples", "checkpoint"},
    "train": _MODEL_OPTIONS | {
        "model_kind", "epochs", "workers", "lr", "energy_weight", "force_weight",
        "max_grad_norm", "checkpoint_every", "validate_every_steps", "shuffle",
        "resume", "max_steps", "progress_seconds",
    },
    "evaluate": {"checkpoint", "split", "device", "batch_size", "workers", "progress_seconds"},
}
_POSITIVE_INT = {
    "max_modes", "atom_basis", "radial_basis", "max_nu", "batch_size", "steps",
    "samples", "epochs", "checkpoint_every", "max_steps",
}
_NONNEGATIVE_INT = {"max_l", "message_passing", "seed", "workers", "validate_every_steps"}
_ENUMS = {
    "dtype": {"float32", "float64"}, "spin_policy": {"exclude", "consistent-state"},
    "model_kind": {"sog", "sr"}, "shuffle": {"affine", "sequential"},
    "split": {"val", "test"},
}
_CONFIG_FIELDS = {
    "stage", "expected_git_commit", "local_work_dir", "remote_project_dir",
    "remote_run_dir", "remote_database", "remote_conda_env", "remote_conda_sh",
    "remote_bundle", "remote_export_dir", "channels", "options", "resume_policy",
    "download_checkpoints", "machine", "resources", "task_name",
}


def _text(value, field):
    if not isinstance(value, str) or not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{field} must be a nonempty string without control characters")
    return value


def _remote_path(value, field):
    value = _text(value, field)
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or "\\" in value or value.startswith("//"):
        raise ValueError(f"{field} must be an absolute Linux path without '..' or backslashes")
    if str(path) == "/":
        raise ValueError(f"{field} must not be the filesystem root")
    return str(path)


def _integer(value, field, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _options(stage, values):
    if not isinstance(values, dict):
        raise ValueError("options must be an object")
    unsupported = set(values) - _OPTIONS[stage]
    if unsupported:
        raise ValueError(f"unsupported options for {stage}: {', '.join(sorted(unsupported))}")
    result = {}
    for key, value in values.items():
        if key in _POSITIVE_INT | _NONNEGATIVE_INT:
            result[key] = _integer(value, key, minimum=0 if key in _NONNEGATIVE_INT else 1)
        elif key in ("checkpoint", "resume"):
            result[key] = _remote_path(value, key)
        elif key in _ENUMS:
            if value not in _ENUMS[key]:
                raise ValueError(f"invalid {key}")
            result[key] = value
        elif key == "n_dl":
            items = value if isinstance(value, list) else [value]
            if not items or (stage != "converge" and len(items) != 1):
                raise ValueError("n_dl must be a positive scalar, or a nonempty list for converge")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in items):
                raise ValueError("n_dl must be finite and positive")
            result[key] = items if stage == "converge" else items[0]
        elif key in {"device", "spin_policy_note"}:
            result[key] = _text(value, key)
            if key == "device" and not re.fullmatch(r"(?:cpu|cuda(?::[0-9]+)?)", value):
                raise ValueError("device must be cpu, cuda, or cuda:<index>")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if value < 0 or (value == 0 and key not in {"energy_weight", "force_weight"}):
                raise ValueError(f"{key} must be positive (loss weights may be zero)")
            result[key] = value
    return result


def _load_config(config):
    if isinstance(config, (str, Path)):
        source = Path(config).expanduser().resolve()
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        base = source.parent
    else:
        value, base = config, Path.cwd()
    if not isinstance(value, dict):
        raise ValueError("dispatch config must be a JSON object")
    unknown = set(value) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unsupported dispatch config fields: {', '.join(sorted(unknown))}")
    required = {"stage", "expected_git_commit", "local_work_dir", "remote_project_dir",
                "remote_run_dir", "remote_database", "remote_conda_env", "machine"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"missing dispatch fields: {', '.join(sorted(missing))}")
    stage = value["stage"]
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}")
    commit = value["expected_git_commit"]
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("expected_git_commit must be a pinned 40-character Git commit SHA")
    local = Path(_text(value["local_work_dir"], "local_work_dir")).expanduser()
    local = (base / local).resolve() if not local.is_absolute() else local.resolve()
    payload = {"stage": stage, "expected_git_commit": commit.lower()}
    for field in ("remote_project_dir", "remote_run_dir", "remote_database", "remote_conda_env"):
        payload[field] = _remote_path(value[field], field)
    for field in ("remote_bundle", "remote_export_dir", "remote_conda_sh"):
        if value.get(field) is not None:
            payload[field] = _remote_path(value[field], field)
    if stage == "bind" and not {"remote_bundle", "remote_export_dir"} <= set(payload):
        raise ValueError("bind requires remote_bundle and remote_export_dir")
    channels = value.get("channels", list(DEFAULT_CHANNELS))
    if not isinstance(channels, list) or not channels:
        raise ValueError("channels must be a nonempty list without duplicates")
    if any(not isinstance(c, str) or not re.fullmatch(r"[A-Za-z0-9_]+", c) for c in channels):
        raise ValueError("invalid channel name")
    if len(set(channels)) != len(channels):
        raise ValueError("channels must not contain duplicates")
    payload["channels"] = channels
    payload["options"] = _options(stage, value.get("options", {}))
    if "spin_review" in channels and stage in {"smoke", "converge", "train"}:
        if payload["options"].get("spin_policy") != "consistent-state" or not payload["options"].get("spin_policy_note"):
            raise ValueError("spin_review requires consistent-state and a reviewed spin_policy_note")
    payload["resume_policy"] = value.get("resume_policy", "auto")
    if payload["resume_policy"] not in {"auto", "never", "required"}:
        raise ValueError("resume_policy must be auto, never, or required")
    if payload["resume_policy"] == "never" and "resume" in payload["options"]:
        raise ValueError("resume_policy=never conflicts with options.resume")
    payload["download_checkpoints"] = value.get("download_checkpoints", False)
    if not isinstance(payload["download_checkpoints"], bool):
        raise ValueError("download_checkpoints must be true or false")
    return value, payload, local


def _machine_resources(config, local):
    machine = config["machine"]
    if not isinstance(machine, dict) or set(machine) - {"remote_root", "remote_profile"}:
        raise ValueError("machine accepts only remote_root and remote_profile")
    profile = machine.get("remote_profile")
    allowed = {"hostname", "username", "port", "key_filename", "timeout", "look_for_keys", "tar_compress"}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise ValueError("remote_profile accepts endpoint and SSH key settings only; keep credentials out of config")
    profile = dict(profile)
    for field in ("hostname", "username"):
        _text(profile.get(field), f"remote_profile.{field}")
    if not re.fullmatch(r"[A-Za-z0-9._:-]+", profile["hostname"]):
        raise ValueError("invalid SSH hostname")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", profile["username"]):
        raise ValueError("invalid SSH username")
    profile["port"] = _integer(profile.get("port"), "SSH port")
    if profile["port"] > 65535:
        raise ValueError("SSH port must be <= 65535")
    if "key_filename" in profile:
        profile["key_filename"] = _text(profile["key_filename"], "key_filename")
    if "timeout" in profile:
        _integer(profile["timeout"], "SSH timeout")
    for field in ("look_for_keys", "tar_compress"):
        if field in profile and not isinstance(profile[field], bool):
            raise ValueError(f"{field} must be true or false")
    settings = config.get("resources", {})
    if not isinstance(settings, dict) or set(settings) - {"cpus_per_task", "memory", "time_limit", "check_interval"}:
        raise ValueError("resources accepts cpus_per_task, memory, time_limit, check_interval")
    cpus = _integer(settings.get("cpus_per_task", 4), "cpus_per_task")
    interval = _integer(settings.get("check_interval", 60), "check_interval")
    flags = [f"#SBATCH --cpus-per-task={cpus}"]
    if settings.get("memory") is not None:
        memory = _text(settings["memory"], "memory")
        if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT])?", memory):
            raise ValueError("memory must be a positive Slurm size such as 32G")
        flags.append(f"#SBATCH --mem={memory}")
    if settings.get("time_limit") is not None:
        limit = _text(settings["time_limit"], "time_limit")
        if not re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", limit):
            raise ValueError("time_limit must use [days-]hours:minutes:seconds")
        flags.append(f"#SBATCH --time={limit}")
    machine_dict = {"batch_type": "Slurm", "context_type": "SSHContext", "local_root": str(local),
                    "remote_root": _remote_path(machine.get("remote_root"), "machine.remote_root"),
                    "remote_profile": profile}
    resources = {"number_node": 1, "cpu_per_node": 1, "gpu_per_node": 1,
                 "queue_name": "qdagnormal", "group_size": 1, "para_deg": 1,
                 "custom_flags": flags}
    return machine_dict, resources, cpus, interval


def _launcher(payload, cpus):
    init = payload.get("remote_conda_sh")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             'task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"']
    if init:
        lines.append("source " + shlex.quote(init))
    else:
        lines += [
            'if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then',
            '  conda_base="$("$CONDA_EXE" info --base)"',
            'elif command -v conda >/dev/null 2>&1; then',
            '  conda_base="$(conda info --base)"',
            'else',
            '  echo "Conda is not on the batch PATH; set remote_conda_sh in the dispatch config." >&2',
            '  exit 2',
            'fi',
            'source "$conda_base/etc/profile.d/conda.sh"',
        ]
    lines += ["conda activate " + shlex.quote(payload["remote_conda_env"]),
              f"export OMP_NUM_THREADS={cpus}", "export PYTHONNOUSERSITE=1",
              "exec " + shlex.quote(payload["remote_conda_env"] + "/bin/python")
              + ' "$task_dir/run_stage.py" "$task_dir/runtime_config.json"']
    return "\n".join(lines) + "\n"


def dispatch(config, submit=False):
    """Create an offline plan; explicitly opt into SSH/Slurm with ``submit=True``.

    ``config`` is a mapping or a JSON path. Relative ``local_work_dir`` is resolved
    against the JSON's directory. No dataset, database, bundle, environment or
    full repository is included in the upload list. The returned plan omits SSH
    authentication details and DPDispatcher's serialized remote profile.
    """
    if not isinstance(submit, bool):
        raise ValueError("submit must be true or false")
    source, payload, local = _load_config(config)
    machine, resources, cpus, interval = _machine_resources(source, local)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    task_name = source.get("task_name", f"{payload['stage']}-{digest}")
    if not isinstance(task_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", task_name):
        raise ValueError("task_name must be a simple name of at most 80 characters")
    payload["task_name"] = task_name
    folder = local / task_name
    folder.mkdir(parents=True, exist_ok=True)
    forward = ["run_stage.sh", "run_stage.py", "runtime_config.json"]
    (folder / "run_stage.sh").write_text(_launcher(payload, cpus), encoding="utf-8", newline="\n")
    (folder / "run_stage.py").write_text(_REMOTE_RUNNER, encoding="utf-8", newline="\n")
    (folder / "runtime_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    plan = {"status": "planned", "submitted": False, "stage": payload["stage"],
            "task_name": task_name, "local_task_dir": str(folder), "forward_files": forward,
            "backward_files": ["results", "stdout.log", "stderr.log"],
            "remote_project_dir": payload["remote_project_dir"],
            "remote_run_dir": payload["remote_run_dir"],
            "remote_database": payload["remote_database"],
            "expected_git_commit": payload["expected_git_commit"], "resources": resources,
            "download_checkpoints": payload["download_checkpoints"],
            "data_transfer": "none; stage data and sealed bundle separately before submitting",
            "cleanup": False, "resume_policy": payload["resume_policy"]}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not submit:
        return plan
    profile = machine["remote_profile"]
    if any("REPLACE" in profile[field].upper() for field in ("hostname", "username")):
        raise ValueError("replace the SSH endpoint placeholders before submitting")
    try:
        from dpdispatcher import Machine, Resources, Submission, Task
    except ImportError as exc:
        raise RuntimeError("DPDispatcher is required only for --submit: install it in the local controller environment") from exc
    remote = Machine.load_from_dict(machine)
    request = Resources.load_from_dict(resources)
    task = Task(command="bash run_stage.sh", task_work_path=task_name, forward_files=forward,
                backward_files=plan["backward_files"], outlog="stdout.log", errlog="stderr.log")
    submission = Submission(work_base=".", machine=remote, resources=request, task_list=[task],
                            forward_common_files=[], backward_common_files=[])
    # Do not return the serialized state: it can include local SSH key settings.
    try:
        submission.run_submission(clean=False, check_interval=interval)
    except Exception:
        # The default fail-fast lifecycle can raise before normal backward-file
        # collection. This pinned DPDispatcher API skips missing files and can
        # retrieve the failed task's already-written diagnostics without cleanup.
        try:
            submission.download_jobs(include_failed=True)
            recovery = "available declared artifacts downloaded; persistent remote files retained"
        except Exception as exc:
            recovery = f"download unavailable ({type(exc).__name__}); persistent remote files retained"
        plan.update(status="failed", submitted=True, recovery=recovery)
        (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    plan.update(status="complete", submitted=True)
    (folder / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


# Sent as a standalone standard-library runner. It is intentionally independent
# of the locally installed adapter version and invokes the pinned remote CLI.
_REMOTE_RUNNER = r'''"""Run one pinned remote stage and collect only declared small artifacts."""
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result_dir = config_path.parent / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    root = Path(config["remote_run_dir"])
    stage = config["stage"]
    stage_dir = root / {"train": "train", "smoke": "smoke", "converge": "converge", "evaluate": "test", "bind": ".dispatch"}[stage]
    marker = root / ".dispatch" / (config["task_name"] + ".json")
    options = dict(config["options"])
    status = {"stage": stage, "status": "failed", "returncode": 1, "started_at": time.time()}
    exit_code = 1
    try:
        project = config["remote_project_dir"]
        actual = subprocess.check_output(["git", "-C", project, "rev-parse", "HEAD"], text=True).strip()
        if actual != config["expected_git_commit"]:
            raise ValueError("remote Git commit does not match expected_git_commit; no stage was executed")
        dirty = subprocess.check_output(["git", "-C", project, "status", "--porcelain", "--untracked-files=no"], text=True)
        if dirty.strip():
            raise ValueError("remote tracked code has local modifications; no stage was executed")
        status["git_commit"] = actual
        if stage == "bind":
            if marker.exists():
                raise ValueError("bind was already attempted by this task; inspect its status and database before rerunning")
            if Path(config["remote_database"]).exists():
                raise ValueError("bind target database already exists; choose the next stage or a new target")
            command = [sys.executable, "-m", "cace_sog_training", "bind", "--bundle", config["remote_bundle"],
                       "--export-dir", config["remote_export_dir"], "--database", config["remote_database"]]
        else:
            if not Path(config["remote_database"]).is_file():
                raise ValueError("remote database is missing; upload and bind the sealed index first")
            command = [sys.executable, "-m", "cace_sog_training", stage, "--database", config["remote_database"],
                       "--channels", *config["channels"]]
            if stage == "train":
                last = stage_dir / "last.pt"
                explicit = options.get("resume")
                policy = config["resume_policy"]
                if explicit and not Path(explicit).is_file():
                    raise ValueError("the explicit resume checkpoint does not exist")
                if policy == "auto" and last.is_file():
                    if explicit and Path(explicit) != last:
                        raise ValueError("auto resume found a different last.pt; choose resume_policy=required explicitly")
                    options["resume"] = str(last)
                elif policy == "required" and not explicit:
                    if not last.is_file():
                        raise ValueError("resume_policy=required but last.pt is missing")
                    options["resume"] = str(last)
                elif marker.exists() and not explicit:
                    raise ValueError("prior training attempt has no resumable checkpoint; refusing to restart from scratch")
                command += ["--run-dir", str(stage_dir)]
            else:
                command += ["--output", str(stage_dir)]
                if stage == "evaluate":
                    options.setdefault("checkpoint", str(root / "train" / "best.pt"))
            for key, value in options.items():
                command.append("--" + key.replace("_", "-"))
                command.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        write_json(marker, {"status": "running", "started_at": status["started_at"], "stage": stage})
        child = subprocess.Popen(command, cwd=project)
        def forward(signum, frame):
            if child.poll() is None:
                child.send_signal(signum)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, forward)
        returncode = child.wait()
        exit_code = 128 - returncode if returncode < 0 else returncode
        status.update(status="complete" if exit_code == 0 else "failed", returncode=exit_code)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        print(status["error"], file=sys.stderr, flush=True)
    finally:
        status["finished_at"] = time.time()
        # Metadata files are bounded in number; checkpoints stay remote unless
        # explicitly requested. No extxyz, bundle or SQLite file is copied.
        names = {"train": ["run_config.json", "run_status.json", "epochs.jsonl", "val_latest.json"],
                 "smoke": ["smoke.json"], "converge": ["convergence.json"],
                 "evaluate": ["metrics.json"], "bind": []}[stage]
        if config["download_checkpoints"]:
            names += {"train": ["last.pt", "best.pt"], "smoke": ["smoke.pt"]}.get(stage, [])
        for name in names:
            source = stage_dir / name
            if source.is_file():
                try:
                    shutil.copy2(source, result_dir / name)
                except OSError as exc:
                    status.setdefault("collection_errors", []).append(f"{name}: {type(exc).__name__}")
                    if exit_code == 0:
                        exit_code = 1
                        status.update(status="failed", returncode=1)
        write_json(result_dir / "dispatch_status.json", status)
        # Only update a marker if execution was attempted previously.
        if marker.exists():
            write_json(marker, status)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))
'''
