"""Create a small, integrity-checked DeePMD debugging workspace.

The official DPA4C trainer samples startup statistics across every system,
so changing only num_steps does not make the full dataset a small smoke test.
This tool copies up to a caller-selected number of existing train and val
systems and only their first set. The original split and exact bytes remain;
no test frames are used. It prepares inputs and never starts training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from .integrity import FORMAT, MAX_LINE, SHA_RE, _hash_file, _identity, _now, _rel

_FILES = ("type.raw", "type_map.raw", "set.000/box.npy", "set.000/coord.npy", "set.000/energy.npy", "set.000/force.npy")


def _read_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("oversized JSON configuration/status")
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object: " + str(path))
    return value


def _write_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _safe_source(root, relative):
    relative = _rel(relative)
    path = root
    for part in relative.split("/"):
        path = path / part
        _identity(path)
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("selected source is not a regular file inside the dataset")
    return resolved


def _choose_systems(path, systems_per_split):
    """Read JSONL once, retaining only the requested tiny system selection."""
    chosen = {"train": [], "val": []}
    digest = hashlib.sha256()
    before = _identity(path, require_file=True)
    last, lines = time.monotonic(), 0
    with path.open("rb") as source:
        while True:
            raw = source.readline(MAX_LINE + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE:
                raise ValueError("systems manifest line exceeds bounded limit")
            lines += 1
            last = _now("Scanned systems", lines, last)
            digest.update(raw)
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("invalid systems manifest record")
            split = row.get("split")
            if split not in chosen or len(chosen[split]) >= systems_per_split:
                continue
            name = _rel(row.get("system"))
            if "/" in name or name in chosen[split]:
                raise ValueError("invalid or duplicate selected system name")
            chosen[split].append(name)
    if _identity(path, require_file=True) != before:
        raise ValueError("systems manifest changed while reading")
    if not all(chosen.values()):
        raise ValueError("both train and val must contain a selectable system")
    return chosen, digest.hexdigest()


def _manifest_files(path, expected_sha, needed, status_sha, systems_sha):
    actual, before = _hash_file(path, label="transfer manifest")
    if actual != expected_sha:
        raise ValueError("transfer manifest SHA256 differs from integrity report")
    digest = hashlib.sha256()
    expected = {}
    line_number, ended, last = 0, False, time.monotonic()
    with Path(path).open("rb") as source:
        while True:
            raw = source.readline(MAX_LINE + 1)
            if not raw:
                break
            if len(raw) > MAX_LINE:
                raise ValueError("transfer manifest line exceeds bounded limit")
            digest.update(raw)
            line_number += 1
            last = _now("Scanned manifest entries", line_number, last)
            row = json.loads(raw.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("invalid transfer manifest record")
            if line_number == 1:
                if row.get("kind") != "header" or row.get("format") != FORMAT or row.get("conversion_status_sha256") != status_sha:
                    raise ValueError("transfer manifest status hash differs from conversion")
                continue
            if ended:
                raise ValueError("records after transfer manifest end")
            if row.get("kind") == "end":
                ended = True
                continue
            if row.get("kind") not in ("file", "dir"):
                raise ValueError("invalid transfer manifest record kind")
            relative = _rel(row.get("path"))
            if relative not in needed:
                continue
            if row.get("kind") != "file" or relative in expected:
                raise ValueError("duplicate or non-file selected entry in transfer manifest")
            sha, size = row.get("sha256"), row.get("size")
            if not isinstance(sha, str) or not SHA_RE.fullmatch(sha) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError("invalid selected file hash or size")
            expected[relative] = (sha, size)
    if digest.hexdigest() != expected_sha or _identity(path, require_file=True) != before or not ended:
        raise ValueError("transfer manifest changed or is incomplete")
    if set(expected) != set(needed):
        raise ValueError("transfer manifest omits a selected smoke file")
    if expected["conversion_status.json"][0] != status_sha or expected["systems.jsonl"][0] != systems_sha:
        raise ValueError("selected systems/status differ from the verified transfer manifest")
    return expected


def _copy_checked(source, destination, expected_sha, expected_size):
    before = _identity(source, require_file=True)
    if before[0] != expected_size:
        raise ValueError("selected file size differs from transfer manifest: " + str(source))
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    length = 0
    with source.open("rb") as inp, destination.open("xb") as out:
        opened = os.fstat(inp.fileno())
        if (int(opened.st_size), int(opened.st_mtime_ns), str(opened.st_dev), str(opened.st_ino)) != before:
            raise ValueError("selected source changed while opening")
        for block in iter(lambda: inp.read(8 * 1024 * 1024), b""):
            out.write(block)
            digest.update(block)
            length += len(block)
    if length != expected_size or digest.hexdigest() != expected_sha or _identity(source, require_file=True) != before:
        raise ValueError("selected source SHA256 mismatch or file changed during copying: " + str(source))
    copied_sha, _ = _hash_file(destination)
    if copied_sha != expected_sha:
        raise ValueError("copied smoke file SHA256 mismatch: " + str(destination))


def prepare_smoke(converted, baseline_config, integrity_report, output, systems_per_split=2, steps=200):
    """Prepare an isolated subset and baseline config without running training."""
    if isinstance(systems_per_split, bool) or not isinstance(systems_per_split, int) or systems_per_split < 1:
        raise ValueError("systems_per_split must be a positive integer")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    root = Path(converted).expanduser().resolve(strict=True)
    config_file = Path(baseline_config).expanduser().resolve(strict=True)
    report_file = Path(integrity_report).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.exists() or os.path.lexists(output):
        raise ValueError("output exists; choose a new directory")
    if output == root or root in output.parents:
        raise ValueError("smoke output must be outside the converted dataset")
    status_path = _safe_source(root, "conversion_status.json")
    status = _read_json(status_path)
    status_sha, _ = _hash_file(status_path)
    report = _read_json(report_file)
    if (status.get("status") != "complete" or status.get("channel") != "sog_candidates"
            or report.get("status") != "complete" or Path(report.get("target_dataset", "")).resolve() != root
            or report.get("conversion_status_sha256") != status_sha):
        raise ValueError("integrity report does not match the complete converted dataset")
    manifest_sha = report.get("expected_manifest_sha256")
    if not isinstance(manifest_sha, str) or not SHA_RE.fullmatch(manifest_sha):
        raise ValueError("integrity report is missing a valid manifest SHA256")
    manifest_path = Path(report.get("manifest", "")).expanduser().absolute()
    resolved_manifest = manifest_path.resolve(strict=True)
    if resolved_manifest == root or root in resolved_manifest.parents:
        raise ValueError("transfer manifest must be outside the dataset")
    config = _read_json(config_file)
    model = config.get("model", {})
    if model.get("descriptor", {}).get("type") != "dpa4c" or "long_range" in model:
        raise ValueError("smoke baseline config must describe plain DPA4C")
    training = config.get("training", {})
    for split, field in (("train", "training_data"), ("val", "validation_data")):
        value = training.get(field, {}).get("systems")
        if not isinstance(value, str) or Path(value).resolve() != root / split:
            raise ValueError("baseline config split path does not match converted dataset: " + split)
    systems_path = _safe_source(root, "systems.jsonl")
    selected, systems_sha = _choose_systems(systems_path, systems_per_split)
    needed = {"conversion_status.json", "systems.jsonl"}
    for split, names in selected.items():
        for name in names:
            needed.update(split + "/" + name + "/" + item for item in _FILES)
    expected = _manifest_files(manifest_path, manifest_sha, needed, status_sha, systems_sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    result = {"status": "partial", "purpose": "debugging only; not the full training dataset",
              "no_training_executed": True, "source_dataset": str(root), "selected_systems": selected,
              "systems_per_split": systems_per_split, "sets_per_system": 1, "steps": steps,
              "source_manifest_sha256": manifest_sha, "source_conversion_status_sha256": status_sha,
              "source_integrity_report_sha256": _hash_file(report_file)[0],
              "source_baseline_config_sha256": _hash_file(config_file)[0]}
    _write_json(output / "smoke_status.json", result)
    try:
        for split, names in selected.items():
            for name in names:
                for item in _FILES:
                    relative = split + "/" + name + "/" + item
                    source = _safe_source(root, relative)
                    sha, size = expected[relative]
                    _copy_checked(source, output / relative, sha, size)
        # Detect metadata changes that happened after the inventory was checked.
        if _hash_file(status_path)[0] != status_sha or _hash_file(systems_path)[0] != systems_sha:
            raise ValueError("source conversion status or system inventory changed during copying")
        smoke = copy.deepcopy(config)
        options = smoke["training"]
        options["training_data"]["systems"] = str(output / "train")
        options["validation_data"]["systems"] = str(output / "val")
        for field in ("stat_file", "stat_file_mode", "numb_epoch", "numb_epochs", "num_epoch", "num_epochs",
                      "numb_step", "numb_steps", "num_step", "num_steps", "stop_batch"):
            options.pop(field, None)
        options["num_steps"] = steps
        options["save_freq"] = min(int(options.get("save_freq", steps)), steps)
        smoke["validating"] = {"full_validation": False, "ema_full_validation": False}
        _write_json(output / "input_dpa4c_smoke.json", smoke)
        result.update(status="complete", config=str(output / "input_dpa4c_smoke.json"))
        _write_json(output / "smoke_status.json", result)
        return result
    except BaseException as exc:
        result.update(status="partial", error=str(exc)[:4096])
        try:
            _write_json(output / "smoke_status.json", result)
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--integrity-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--systems-per-split", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        result = prepare_smoke(args.converted, args.baseline_config, args.integrity_report,
                               args.output, args.systems_per_split, args.steps)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; smoke workspace is partial.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
