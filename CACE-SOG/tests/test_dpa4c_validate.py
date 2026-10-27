"""Full held-out validation selection without a real DeePMD installation."""
import json
from pathlib import Path

import numpy as np
import pytest

from cace_sog_training.data import IndexReader, build_index
from dpa4c_sog.validate import validate
from elementa_fixtures import exported_fixture
from test_dpa4c_predict import FakeDeepPot


class ScaleModel(FakeDeepPot):
    def __init__(self, kind, scale):
        super().__init__(kind)
        self.scale = scale

    def eval(self, coords, cells, atom_types, atomic=False, charge_spin=None):
        energy, forces, virial = super().eval(coords, cells, atom_types, atomic, charge_spin)
        forces[0, 0, 0] = self.scale
        forces[0, 1, 0] = -self.scale
        return energy, forces, virial


def inputs(tmp_path, *, val=3):
    export = tmp_path / "export"
    exported_fixture(export, train=2, val=val, test=2)
    database = tmp_path / "index.sqlite3"
    build_index(export, database)
    files = [tmp_path / "candidate1.pt", tmp_path / "candidate2.pt"]
    for i, file in enumerate(files):
        file.write_bytes(("candidate%d" % i).encode())
    return database, files


def test_selects_best_using_all_val_frames_and_never_test(tmp_path, monkeypatch):
    database, files = inputs(tmp_path)
    models = {files[0]: ScaleModel("dpa", 1), files[1]: ScaleModel("dpa", 0)}
    original = IndexReader.iter_records
    seen_splits = []
    def only_val(self, channels, split):
        seen_splits.append(split)
        assert tuple(channels) == ("sog_candidates",) and split == "val"
        yield from original(self, channels, split)
    monkeypatch.setattr(IndexReader, "iter_records", only_val)
    report = validate(database, "dpa", files, tmp_path / "report", (0, 1), loader=models.get)
    assert report["selected_index"] == 1
    assert report["selected_checkpoint"] == str(files[1])
    assert report["frames_per_checkpoint"] == 3 and not report["test_split_read"]
    assert seen_splits == ["val", "val"]
    for row in report["candidates"]:
        assert row["energy"]["count"] == 3 and row["force"]["count"] == 18
        assert len(row["checkpoint_sha256"]) == 64
    assert report["candidates"][1]["force"]["mae"] < report["candidates"][0]["force"]["mae"]
    assert json.loads((tmp_path / "report/validation_status.json").read_text())["status"] == "complete"
    assert json.loads((tmp_path / "report/validation.json").read_text())["index_identity"] == report["index_identity"]


def test_exact_ties_prefer_first_candidate_and_sog_metadata(tmp_path):
    database, files = inputs(tmp_path)
    models = {p: ScaleModel("sog", 0) for p in files}
    report = validate(database, "sog", files, tmp_path / "report", (0, 1), loader=models.get)
    assert report["selected_index"] == 0 and report["selected_checkpoint"] == str(files[0])


def test_empty_val_and_bad_model_fail_without_completed_report(tmp_path):
    database, files = inputs(tmp_path, val=0)
    with pytest.raises(ValueError, match="split is empty"):
        validate(database, "dpa", files, tmp_path / "empty", (0, 1), loader=lambda _: ScaleModel("dpa", 0))
    assert json.loads((tmp_path / "empty/validation_status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "empty/validation.json").exists()
    with pytest.raises(ValueError, match="without a long-range"):
        validate(database, "dpa", files, tmp_path / "wrong", (0, 1), loader=lambda _: ScaleModel("sog", 0))
    assert not (tmp_path / "wrong/validation.json").exists()


def test_changed_source_is_rejected_between_checkpoints(tmp_path):
    database, files = inputs(tmp_path)
    def load(path):
        if path == files[1]:
            with (tmp_path / "export/sog_candidates/val.extxyz").open("ab") as out:
                out.write(b"changed")
        return ScaleModel("dpa", 0)
    with pytest.raises(ValueError, match="changed"):
        validate(database, "dpa", files, tmp_path / "report", (0, 1), loader=load)
    status = json.loads((tmp_path / "report/validation_status.json").read_text())
    assert status["status"] == "failed" and status["candidates_completed"] == 1
    assert not (tmp_path / "report/validation.json").exists()


def test_changed_index_rows_detected_even_when_metadata_is_unchanged(tmp_path):
    import sqlite3
    database, files = inputs(tmp_path)
    def load(path):
        if path == files[1]:
            with sqlite3.connect(str(database)) as db:
                rows = db.execute("SELECT id,offset,length,sha256 FROM records WHERE channel='sog_candidates' AND split='val' ORDER BY ordinal LIMIT 2").fetchall()
                db.execute("UPDATE records SET offset=?,length=?,sha256=? WHERE id=?", (*rows[1][1:], rows[0][0]))
                db.execute("UPDATE records SET offset=?,length=?,sha256=? WHERE id=?", (*rows[0][1:], rows[1][0]))
        return ScaleModel("dpa", 0)
    with pytest.raises(ValueError, match="frame identities changed"):
        validate(database, "dpa", files, tmp_path / "report", (0, 1), loader=load)
    assert not (tmp_path / "report/validation.json").exists()


def test_rejects_overwrite_and_inconsistent_assumption(tmp_path):
    database, files = inputs(tmp_path)
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError, match="exists"):
        validate(database, "dpa", files, tmp_path / "existing", (0, 1), loader=lambda _: None)
    with pytest.raises(ValueError, match="0 1"):
        validate(database, "dpa", files, tmp_path / "new", (0, 3), loader=lambda _: None)


def test_checkpoint_changed_mid_validation_rejected(tmp_path):
    database, files = inputs(tmp_path)
    class MutatingModel(ScaleModel):
        def eval(self, coords, cells, atom_types, atomic=False, charge_spin=None):
            result = super().eval(coords, cells, atom_types, atomic, charge_spin)
            files[0].write_bytes(b"checkpoint-mutated")
            return result
    with pytest.raises(ValueError, match="checkpoint changed"):
        validate(database, "dpa", files[:1], tmp_path / "report", (0, 1), loader=lambda _: MutatingModel("dpa", 0))
    assert not (tmp_path / "report/validation.json").exists()
