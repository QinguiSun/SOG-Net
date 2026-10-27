"""Tiny index-based smoke preparation never touches held-out test frames."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ase.data import chemical_symbols
import numpy as np
import pytest

from cace_sog_training.data import IndexReader, build_index
from dpa4c_sog.index_smoke import MAX_FRAMES_PER_SPLIT, prepare_index_smoke
from test_dpa4c_convert import make_dataset


def release_file(path, **descriptor_changes):
    config = {"model": {"type_map": chemical_symbols[1:119],
                        "descriptor": {"type": "dpa4c", "rcut": 6.0,
                                       "add_chg_spin_ebd": True, "default_chg_spin": [0, 1]},
                        "fitting_net": {"neuron": [256, 256, 256]}},
              "training": {"stat_file": "old-OMol.hdf5", "num_epochs": 25, "zero_stage": 1},
              "learning_rate": {"type": "wsd", "start_lr": 0.003, "stop_lr": 1e-6}}
    config["model"]["descriptor"].update(descriptor_changes)
    path.write_text(json.dumps(config), encoding="utf-8")
    return config


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tiny_dataset_reorders_coordinates_and_forces_and_preserves_source(tmp_path, monkeypatch):
    source, database = make_dataset(tmp_path)
    release = tmp_path / "released.json"
    original_config = release_file(release)
    original_files = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()}
    index_hash = digest(database)
    config_hash = digest(release)
    seen_splits = []
    original_iter = IndexReader.iter_records
    def train_val_only(self, channels, split):
        seen_splits.append(split)
        assert tuple(channels) == ("sog_candidates",) and split in ("train", "val")
        yield from original_iter(self, channels, split)
    monkeypatch.setattr(IndexReader, "iter_records", train_val_only)
    out = tmp_path / "small-workspace"
    result = prepare_index_smoke(database, release, out, [0, 1], frames_per_split=2, steps=7)
    assert result["status"] == "complete" and result["debug_only"] is True
    assert result["no_training_executed"] and not result["test_split_read"]
    assert result["counts"] == {"train": 2, "val": 2}
    assert seen_splits == ["train", "val"]
    assert not (out / "test").exists()
    assert digest(database) == index_hash and digest(release) == config_hash
    assert {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()} == original_files

    config = json.loads((out / "input_dpa4c_smoke.json").read_text(encoding="utf-8"))
    assert config["model"] == original_config["model"]
    assert config["training"]["training_data"]["systems"] == str(out / "train")
    assert config["training"]["validation_data"]["systems"] == str(out / "val")
    assert config["training"]["num_steps"] == config["training"]["save_freq"] == 7
    assert config["validating"]["full_validation"] is False
    assert "stat_file" not in config["training"] and "num_epochs" not in config["training"]
    assert str(out / "test") not in json.dumps(config["training"])
    assert result["frames_manifest_sha256"] == digest(out / "frames.jsonl")
    assert result["config_sha256"] == digest(out / "input_dpa4c_smoke.json")
    assert result["data_inventory_sha256"] == digest(out / "smoke_files.json")
    inventory = json.loads((out / "smoke_files.json").read_text(encoding="utf-8"))["files"]
    for name, item in inventory.items():
        assert digest(out / name) == item["sha256"]

    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {r["split"] for r in rows} == {"train", "val"}
    with IndexReader(database) as reader:
        for item in rows:
            record = reader.record("sog_candidates", item["split"], item["source_ordinal"])
            original = reader.read_atoms(record)
            assert item["source_id"] == record["id"] and item["frame_sha256"] == record["sha256"]
            folder = out / item["split"] / item["system"]
            subset = folder / "set.000"
            order = item["canonical_to_source"]
            assert (folder / "type.raw").read_text().split() == ["2", "8"]
            assert (folder / "type_map.raw").read_text().split() == chemical_symbols[1:119]
            np.testing.assert_array_equal(np.load(subset / "coord.npy")[item["row"]].reshape(-1, 3), original.positions[order])
            np.testing.assert_array_equal(np.load(subset / "force.npy")[item["row"]].reshape(-1, 3), original.arrays["REF_forces"][order])
            np.testing.assert_array_equal(np.load(subset / "box.npy")[item["row"]].reshape(3, 3), original.cell.array)
            assert np.load(subset / "energy.npy")[item["row"]] == original.info["REF_energy"]


def test_request_more_frames_than_available_keeps_all_available(tmp_path):
    _, database = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    release_file(release)
    status = prepare_index_smoke(database, release, tmp_path / "out", [0, 1], frames_per_split=4)
    assert status["counts"] == {"train": 3, "val": 2}
    assert len(status["source_frames"]) == 5


def test_explicit_conflicting_electronic_state_leaves_partial_result(tmp_path):
    _, database = make_dataset(tmp_path, invalid=True)
    release = tmp_path / "release.json"
    release_file(release)
    output = tmp_path / "partial"
    with pytest.raises(ValueError, match="conflicts"):
        prepare_index_smoke(database, release, output, [0, 1], frames_per_split=2)
    status = json.loads((output / "smoke_status.json").read_text())
    assert status["status"] == "partial" and "conflicts" in status["error"]
    assert not (output / "input_dpa4c_smoke.json").exists()
    assert not list(output.glob(".index-smoke-spool-*"))


@pytest.mark.parametrize("failure", ["condition", "count", "steps", "release", "existing", "source", "partial"])
def test_invalid_inputs_never_publish_complete_result(tmp_path, failure):
    source, database = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    release_file(release, type="not-dpa4c" if failure == "release" else "dpa4c")
    target = tmp_path / "result"
    if failure == "existing":
        target.mkdir()
        (target / "owned").write_text("unchanged")
    if failure == "source":
        target = source / "new-output"
    if failure == "partial":
        database = tmp_path / "partial.sqlite3"
        build_index(source, database, limit_records=1)
    with pytest.raises(ValueError):
        prepare_index_smoke(database, release, target, [0, 3] if failure == "condition" else [0, 1],
            frames_per_split=MAX_FRAMES_PER_SPLIT + 1 if failure == "count" else 4,
            steps=0 if failure == "steps" else 200)
    assert not (target / "smoke_status.json").exists()
    if failure == "existing":
        assert (target / "owned").read_text() == "unchanged"


def test_preexisting_dangling_symlink_is_not_followed(tmp_path):
    _, database = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    release_file(release)
    target = tmp_path / "link"
    try:
        target.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is unavailable")
    with pytest.raises(ValueError, match="already exists"):
        prepare_index_smoke(database, release, target, [0, 1])
    assert not (tmp_path / "does-not-exist").exists()


def test_cli_executes_without_torch_or_deepmd(tmp_path):
    _, database = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    release_file(release)
    root = Path(__file__).resolve().parents[1]
    code = '''
import importlib.abc, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'deepmd'):
            raise RuntimeError('unexpected ML import ' + fullname)
sys.meta_path.insert(0, NoML())
from dpa4c_sog.index_smoke import main
sys.exit(main(sys.argv[1:]))
'''
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONUTF8="1")
    output = tmp_path / "cli"
    process = subprocess.run([sys.executable, "-X", "utf8", "-c", code,
        "--database", str(database), "--release-config", str(release), "--output", str(output),
        "--assume-charge-spin", "0", "1", "--frames-per-split", "1", "--steps", "3"],
        cwd=root, env=env, capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    status = json.loads(process.stdout)
    assert status["status"] == "complete" and status["debug_only"]
    assert status["counts"] == {"train": 1, "val": 1}
