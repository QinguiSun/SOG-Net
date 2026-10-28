"""Owned fixtures verify frame identity, canonical ordering and failed status."""
import json
from pathlib import Path

import numpy as np
import pytest

from cace_sog_training.data import build_index, IndexReader
from dpa4c_sog.convert import TYPE_MAP, convert
from test_elementa_data import fixture, frame


def reversed_frame(energy=-2):
    return ("2\n"
            'Lattice="8 0 0 0 8 0 0 0 8" Properties=species:S:1:pos:R:3:forces:R:3 pbc="T T T" energy=%s\n'
            "F 1.7 0 0 -0.1 0 0\nLi 0 0 0 0.1 0 0\n" % energy).encode()


def make_dataset(base, invalid=False):
    source = base / "export"
    first = frame(-4)
    third = frame(-3)
    if invalid:
        third = frame(-3, metadata={"charge_spin": '"0 3"'})
    fixture(source, [
        ("sog_candidates", "train", "train-1", first),
        ("sog_candidates", "train", "train-2", reversed_frame(-2)),
        ("sog_candidates", "train", "train-3", first),  # true duplicates retained
        ("sog_candidates", "val", "val-1", third),
        ("sog_candidates", "val", "val-2", reversed_frame(-1)),
        ("sog_candidates", "test", "test-1", frame(0)),
        ("lattice_auxiliary", "train", "other-1", frame(-9)),
        ("lattice_auxiliary", "val", "other-2", frame(-9)),
        ("lattice_auxiliary", "test", "other-3", frame(-9)),
    ])
    db = base / "training.sqlite3"
    build_index(source, db, commit_every=2)
    return source, db


def test_splits_atom_order_total_energy_force_mapping_and_no_sampling(tmp_path):
    source, db = make_dataset(tmp_path)
    out = tmp_path / "deepmd"
    result = convert(db, out, [0, 1], set_frames=2, set_bytes=65536, commit_every=2)
    assert result["status"] == "complete"
    assert result["counts"] == {"train": 3, "val": 2, "test": 1}
    assert result["no_sampling"] is True
    assert result["assumed_charge_spin"] == [0, 1]
    assert result["charge_spin_is_model_input_assumption_not_verified_dft_state"]
    assert not (out / ".conversion.sqlite3").exists()
    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert len({row["source_id"] for row in rows}) == 6
    assert len({row["sha256"] for row in rows}) < 6  # duplicate bytes stay as distinct examples
    assert not any(row["channel"] != "sog_candidates" for row in rows)
    assert len(TYPE_MAP) == 118 and TYPE_MAP[2] == "Li" and TYPE_MAP[8] == "F"
    with IndexReader(db) as reader:
        for item in rows:
            record = reader.record("sog_candidates", item["split"], item["source_ordinal"])
            original = reader.read_atoms(record)
            assert record["id"] == item["source_id"] and record["sha256"] == item["sha256"]
            folder = out / item["split"] / item["system"]
            assert (folder / "type_map.raw").read_text().split() == list(TYPE_MAP)
            assert (folder / "type.raw").read_text().split() == ["2", "8"]
            subset = folder / item["set"]
            coords = np.load(subset / "coord.npy")
            forces = np.load(subset / "force.npy")
            energy = np.load(subset / "energy.npy")
            box = np.load(subset / "box.npy")
            assert coords.shape[0] <= 2
            order = item["canonical_to_source"]
            np.testing.assert_array_equal(coords[item["row"]].reshape(-1, 3), original.positions[order])
            np.testing.assert_array_equal(forces[item["row"]].reshape(-1, 3), original.arrays["REF_forces"][order])
            np.testing.assert_array_equal(box[item["row"]].reshape(3, 3), original.cell.array)
            assert energy[item["row"]] == original.info["REF_energy"]
    assert {row["set"] for row in rows if row["split"] == "train"} == {"set.000", "set.001"}


def test_byte_budget_limits_single_frame_sets(tmp_path):
    _, db = make_dataset(tmp_path)
    out = tmp_path / "deepmd"
    convert(db, out, [0, 1], set_frames=999, set_bytes=1)
    rows = [json.loads(line) for line in (out / "frames.jsonl").read_text().splitlines()]
    assert len({r["set"] for r in rows if r["split"] == "train"}) == 3


def test_incompatible_explicit_condition_leaves_partial_status(tmp_path):
    _, db = make_dataset(tmp_path, invalid=True)
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="conflicts"):
        convert(db, out, [0, 1])
    status = json.loads((out / "conversion_status.json").read_text())
    assert status["status"] == "partial"
    assert "conflicts" in status["error"]


def test_no_overwrite_no_implicit_spin_and_complete_index_required(tmp_path):
    source, db = make_dataset(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    (out / "owned").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        convert(db, out, [0, 1])
    assert (out / "owned").read_text() == "keep"
    with pytest.raises(ValueError, match="0 1"):
        convert(db, tmp_path / "wrong-condition", [0, 3])
    partial = tmp_path / "partial.sqlite3"
    build_index(source, partial, limit_records=1)
    with pytest.raises(ValueError, match="complete index"):
        convert(partial, tmp_path / "partial-out", [0, 1])
    assert not (tmp_path / "partial-out").exists()
