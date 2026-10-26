"""The portable fit uses train only and remains independent of the model stack."""
import json

import pytest

from cace_sog_training.baseline import fit_baseline, prepared_baseline
from cace_sog_training.data import build_index, IndexReader
from test_elementa_data import fixture, frame


def test_baseline_ignores_validation_and_test_labels(tmp_path):
    root = tmp_path / "export"
    fixture(root, [("sog_candidates", "train", "a", frame(-4)),
                   ("sog_candidates", "val", "b", frame(10000)),
                   ("sog_candidates", "test", "c", frame(-10000))])
    db = tmp_path / "index.sqlite3"
    build_index(root, db, channels=["sog_candidates"])
    with IndexReader(db) as reader:
        offsets, fit = fit_baseline(reader, ["sog_candidates"])
        assert sum(offsets.values()) == pytest.approx(-4)
        assert fit["frames"] == 1
        assert fit["normal_matrix_rank"] == 1
        assert prepared_baseline(reader, ["sog_candidates"]) is None
        reader.metadata["prepared_baseline"] = dict(
            channels=["sog_candidates"], e0=json.loads(json.dumps(offsets)), fit=fit)
        assert prepared_baseline(reader, ["sog_candidates"]) == (offsets, fit)
        assert prepared_baseline(reader, ["lattice_auxiliary"]) is None
        reader.metadata["prepared_baseline"]["e0"]["3"] = float("nan")
        with pytest.raises(ValueError, match="invalid prepared"):
            prepared_baseline(reader, ["sog_candidates"])
