import json
from pathlib import Path

import numpy as np
import pytest

from cace_sog_training.data import IndexReader
from dpa4c_sog.stratified_pilot import prepare_pilot, select_records
from test_dpa4c_convert import make_dataset
from test_dpa4c_index_smoke import release_file


def test_element_sampling_bounded_deterministic_and_order_independent():
    records = [dict(id=i, ordinal=i, component_id=f"component-{i // 5}",
                    counts_json=json.dumps({str(1 + i % 8): 1, "11": 2}))
               for i in range(1000)]
    first, elements, seen = select_records(iter(records), per_element=2, seed=31, progress_seconds=0)
    second, _, _ = select_records(iter(reversed(records)), per_element=2, seed=31, progress_seconds=0)
    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert len(first) <= 9 * 2
    assert elements == list(range(1, 9)) + [11]
    assert seen == 1000


def test_long_trajectory_does_not_crowd_out_another_component():
    rows = [dict(id=i, ordinal=i, component_id="long", counts_json='{"11": 1}')
            for i in range(1000)]
    rows.append(dict(id=1000, ordinal=1000, component_id="short", counts_json='{"11": 1}'))
    result, _, _ = select_records(rows, per_element=2, progress_seconds=0)
    assert {row["component_id"] for row in result} == {"long", "short"}
    assert len(result) == 2


def test_pilot_preserves_splits_labels_and_source(tmp_path, monkeypatch):
    source, db = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    original = release_file(release)
    observed = []
    iter_records = IndexReader.iter_records

    def only_train_val(self, channels, split):
        assert split in ("train", "val")
        observed.append(split)
        yield from iter_records(self, channels, split)

    monkeypatch.setattr(IndexReader, "iter_records", only_train_val)
    target = tmp_path / "pilot"
    result = prepare_pilot(db, release, target, per_element=2, seed=7, steps=9)
    assert result["status"] == "complete"
    assert result["test_split_read"] is False
    assert observed == ["train", "val"]
    assert not (target / "test").exists()
    assert result["train_composition_rank"] <= result["train_element_count"]
    cfg = json.loads((target / "input_dpa4c_pilot.json").read_text())
    assert cfg["model"] == original["model"]
    assert cfg["training"]["num_steps"] == 9
    rows = [json.loads(line) for line in (target / "frames.jsonl").read_text().splitlines()]
    assert len(rows) == sum(result["counts"].values())
    with IndexReader(db) as reader:
        for item in rows:
            record = reader.record("sog_candidates", item["split"], item["source_ordinal"])
            atoms = reader.read_atoms(record)
            arrays = target / item["split"] / item["system"] / item["set"]
            order = item["canonical_to_source"]
            np.testing.assert_array_equal(np.load(arrays / "coord.npy")[item["row"]].reshape(-1, 3), atoms.positions[order])
            np.testing.assert_array_equal(np.load(arrays / "force.npy")[item["row"]].reshape(-1, 3), atoms.arrays["REF_forces"][order])
            assert np.load(arrays / "energy.npy")[item["row"]] == atoms.info["REF_energy"]
    assert source.exists()


def test_rejects_invalid_quota_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="per-element"):
        select_records([], per_element=0)
    _, db = make_dataset(tmp_path)
    release = tmp_path / "release.json"
    release_file(release)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        prepare_pilot(db, release, output)
