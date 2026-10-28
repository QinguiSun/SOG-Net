"""Small owned systems exercise the safe smoke subset and provenance gates."""
import hashlib
import json
from pathlib import Path

import pytest

from cace_sog_training.data import build_index
from dpa4c_sog.convert import convert
from dpa4c_sog.integrity import seal, verify
from dpa4c_sog.smoke import _FILES, prepare_smoke
from test_elementa_data import fixture


def structure(a, b, energy):
    return ("2\n"
            'Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T" energy=%s\n'
            "%s 0 0 0 0.1 0 0\n%s 1.7 0 0 -0.1 0 0\n" % (energy, a, b)).encode()


def inputs(tmp_path):
    source = tmp_path / "export"
    data = []
    for split in ("train", "val", "test"):
        for family, (a, b) in enumerate((("Li", "F"), ("Na", "Cl"), ("Mg", "O"))):
            for n in range(2):
                data.append(("sog_candidates", split, f"{split}-{family}-{n}", structure(a, b, -2 + n / 10)))
    fixture(source, data)
    index = tmp_path / "index.sqlite3"
    build_index(source, index, channels=["sog_candidates"])
    converted = tmp_path / "converted"
    convert(index, converted, [0, 1], set_frames=1)
    manifest = tmp_path / "manifest.jsonl"
    sealed = seal(converted, manifest)
    report = tmp_path / "verified.json"
    verify(converted, manifest, sealed["manifest_sha256"], report)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"model": {"descriptor": {"type": "dpa4c"}, "some_parameter": 42},
        "training": {"training_data": {"systems": str(converted / "train")},
                     "validation_data": {"systems": str(converted / "val")},
                     "num_steps": 500, "save_freq": 400, "stat_file": "old.hdf5"},
        "validating": {"full_validation": True}}))
    return converted, baseline, report


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_copies_only_small_original_split_subset_and_preserves_model(tmp_path):
    source, baseline, report = inputs(tmp_path)
    old_status = (source / "conversion_status.json").read_bytes()
    out = tmp_path / "smoke"
    result = prepare_smoke(source, baseline, report, out, systems_per_split=2, steps=200)
    assert result["status"] == "complete" and result["no_training_executed"]
    assert len(result["selected_systems"]["train"]) == 2
    assert len(result["selected_systems"]["val"]) == 2
    assert not (out / "test").exists()
    for split, names in result["selected_systems"].items():
        for name in names:
            for item in _FILES:
                assert sha(out / split / name / item) == sha(source / split / name / item)
            assert not (out / split / name / "set.001").exists()
    config = json.loads((out / "input_dpa4c_smoke.json").read_text())
    original = json.loads(baseline.read_text())
    assert config["model"] == original["model"]
    assert config["training"]["training_data"]["systems"] == str(out / "train")
    assert config["training"]["validation_data"]["systems"] == str(out / "val")
    assert config["training"]["num_steps"] == 200 and config["training"]["save_freq"] == 200
    assert "stat_file" not in config["training"]
    assert config["validating"]["full_validation"] is False
    assert (source / "conversion_status.json").read_bytes() == old_status
    with pytest.raises(ValueError, match="exists"):
        prepare_smoke(source, baseline, report, out)
    assert json.loads((out / "smoke_status.json").read_text())["status"] == "complete"


def test_altered_selected_file_fails_and_workspace_remains_partial(tmp_path):
    source, baseline, report = inputs(tmp_path)
    records = [json.loads(line) for line in (source / "systems.jsonl").read_text().splitlines()]
    chosen = next(row["system"] for row in records if row["split"] == "train")
    path = source / "train" / chosen / "set.000" / "coord.npy"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_smoke(source, baseline, report, out)
    assert json.loads((out / "smoke_status.json").read_text())["status"] == "partial"
    assert not (out / "input_dpa4c_smoke.json").exists()


def test_changed_system_inventory_or_wrong_config_is_rejected(tmp_path):
    source, baseline, report = inputs(tmp_path)
    inventory = source / "systems.jsonl"
    original = inventory.read_bytes()
    inventory.write_bytes(original + b'{}\n')
    with pytest.raises(ValueError, match="differ from"):
        prepare_smoke(source, baseline, report, tmp_path / "bad-inventory")
    inventory.write_bytes(original)
    value = json.loads(baseline.read_text())
    value["training"]["training_data"]["systems"] = str(source / "test")
    baseline.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="split path"):
        prepare_smoke(source, baseline, report, tmp_path / "bad-config")


def test_wrong_report_or_output_inside_source_rejected(tmp_path):
    source, baseline, report = inputs(tmp_path)
    content = json.loads(report.read_text())
    content["target_dataset"] = str(tmp_path / "different")
    report.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="integrity report"):
        prepare_smoke(source, baseline, report, tmp_path / "bad-report")
    with pytest.raises(ValueError, match="outside"):
        prepare_smoke(source, baseline, report, source / "unsafe")
