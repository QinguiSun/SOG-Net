"""Paired checkpoint prediction contract, exercised without a DeePMD install."""
import json
from pathlib import Path

import numpy as np
import pytest

from cace_sog_training.data import build_index
from dpa4c_sog.predict import predict
from elementa_fixtures import exported_fixture


class FakeDeepPot:
    def __init__(self, kind, *, bad_shape=False, nonfinite=False, conditioned=True):
        self.kind = kind
        # Distinct type ordering tests per-model mapping and atom-order retention.
        self.types = ["Na", "Cl"] if kind == "dpa" else ["Cl", "Na"]
        self.deep_eval = self
        self.bad_shape, self.nonfinite, self.conditioned = bad_shape, nonfinite, conditioned
        self.calls = []

    def get_type_map(self):
        return self.types

    def has_chg_spin_ebd(self):
        return self.conditioned

    def get_dim_fparam(self):
        return 0

    def get_dim_aparam(self):
        return 0

    def get_model_def_script(self):
        result = {"descriptor": {"type": "dpa4c"}}
        if self.kind == "sog":
            result["long_range"] = {"type": "latent_charge", "kernel": {"type": "sog"}, "periodic": True}
        return result

    def eval(self, coords, cells, atom_types, atomic=False, charge_spin=None):
        assert not atomic
        assert np.array_equal(charge_spin, [[0, 1]])
        assert np.asarray(coords).shape == (1, 2, 3)
        assert np.asarray(cells).shape == (1, 9)
        assert [self.types[int(i)] for i in atom_types] == ["Na", "Cl"]
        self.calls.append((coords.copy(), cells.copy(), atom_types.copy(), charge_spin.copy()))
        base = 1.0 if self.kind == "dpa" else 2.0
        e = np.array([[base]])
        f = np.array([[[base, 0, 0], [-base, 0, 0]]])
        if self.bad_shape:
            f = f.reshape(2, 3)
        if self.nonfinite:
            f[0, 0, 0] = np.nan
        return e, f, np.zeros((1, 9))


@pytest.fixture
def paired_inputs(tmp_path):
    root = tmp_path / "export"
    exported_fixture(root, train=2, val=1, test=2)
    database = tmp_path / "index.sqlite3"
    build_index(root, database)
    dpa, sog = tmp_path / "dpa.pt", tmp_path / "sog.pt"
    dpa.write_bytes(b"owned-test-dpa-checkpoint")
    sog.write_bytes(b"owned-test-sog-checkpoint")
    return database, dpa, sog


def run(tmp_path, paired_inputs, dpa_model=None, sog_model=None, condition=(0, 1)):
    db, dpa, sog = paired_inputs
    dpa_model = dpa_model or FakeDeepPot("dpa")
    sog_model = sog_model or FakeDeepPot("sog")
    models = {dpa: dpa_model, sog: sog_model}
    from dpa4c_sog.validate import validate
    reports = {"dpa": tmp_path / "dpa-validation" / "validation.json",
               "sog": tmp_path / "sog-validation" / "validation.json"}
    for label, checkpoint in (("dpa", dpa), ("sog", sog)):
        if not reports[label].exists():
            validate(db, label, [checkpoint], reports[label].parent, (0, 1),
                     loader=lambda unused: FakeDeepPot(label), progress_seconds=1000)
    path = tmp_path / "paired.jsonl"
    result = predict(db, dpa, sog, path, condition, validation_reports=reports,
                     loader=lambda p: models[p], progress_seconds=1000)
    return path, result, dpa_model, sog_model


def test_complete_pairwise_predictions_and_provenance(tmp_path, paired_inputs):
    path, result, dpa, sog = run(tmp_path, paired_inputs)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == result["expected_frames"] == result["frames_written"] == 2
    assert [r["frame_key"] for r in rows] == ["sog_candidates/test/0", "sog_candidates/test/1"]
    assert all(r["split"] == "test" and len(r["frame_sha256"]) == 64 and r["natoms"] == 2 for r in rows)
    assert rows[0]["dpa_forces_eV_A"] == [[1, 0, 0], [-1, 0, 0]]
    assert rows[0]["sog_forces_eV_A"] == [[2, 0, 0], [-2, 0, 0]]
    assert len(dpa.calls) == len(sog.calls) == 2
    assert np.array_equal(dpa.calls[0][0], sog.calls[0][0])
    assert np.array_equal(dpa.calls[0][1], sog.calls[0][1])
    status = json.loads(path.with_name(path.name + ".status.json").read_text())
    assert status["status"] == "complete" and len(status["predictions_sha256"]) == 64
    assert status["models"]["dpa"]["type_map"] != status["models"]["sog"]["type_map"]
    assert status["condition_provenance"].startswith("explicit constant user assumption")
    assert len(status["index_identity"]) == 64
    assert status["checkpoints"]["dpa"]["sha256"] != status["checkpoints"]["sog"]["sha256"]
    assert status["fine_tune_parent_not_verified"] is True
    assert status["architecture_comparison"]["status"] == "partially_compared"
    assert "descriptor.channels" in status["architecture_comparison"]["not_compared_fields"]
    assert status["validation_provenance"]["dpa"]["frames"] == 1
    assert status["validation_provenance"]["dpa"]["frame_identity_sha256"] == status["validation_provenance"]["sog"]["frame_identity_sha256"]
    with pytest.raises(ValueError, match="already exists"):
        run(tmp_path, paired_inputs)


def test_prediction_file_feeds_full_paired_metrics_and_plots(tmp_path, paired_inputs):
    from dpa4c_sog.metrics import evaluate_predictions
    path, _, _, _ = run(tmp_path, paired_inputs)
    result = evaluate_predictions(path, tmp_path / "report", bins=16)
    assert result["paired_frames"] == 2
    assert result["metrics"]["dpa"]["energy"]["count"] == 2
    assert result["metrics"]["sog"]["force"]["count"] == 12
    assert (tmp_path / "report/dpa_parity.png").is_file()
    assert (tmp_path / "report/sog_parity.png").is_file()


@pytest.mark.parametrize("bad", [
    FakeDeepPot("sog", bad_shape=True),
    FakeDeepPot("sog", nonfinite=True),
    FakeDeepPot("sog", conditioned=False),
])
def test_failure_never_publishes_result(tmp_path, paired_inputs, bad):
    with pytest.raises(ValueError):
        run(tmp_path, paired_inputs, sog_model=bad)
    path = tmp_path / "paired.jsonl"
    assert not path.exists()
    assert json.loads(path.with_name(path.name + ".status.json").read_text())["status"] == "failed"


def test_type_map_mismatch_and_unconditioned_backend_fail(tmp_path, paired_inputs):
    model = FakeDeepPot("sog")
    model.types = ["Na", "O"]
    with pytest.raises(ValueError, match="different elements"):
        run(tmp_path, paired_inputs, sog_model=model)
    assert not (tmp_path / "paired.jsonl").exists()


class ConfiguredDeepPot(FakeDeepPot):
    def __init__(self, kind, descriptor=None, neuron=None):
        super().__init__(kind)
        self.extra_descriptor = {"channels": 64, "lmax": 2, "radial_modes": 0, "rcut": 6.0}
        self.extra_descriptor.update(descriptor or {})
        self.neuron = [256, 256, 256] if neuron is None else neuron

    def get_model_def_script(self):
        result = super().get_model_def_script()
        result["descriptor"].update(self.extra_descriptor)
        result["fitting_net"] = {"neuron": self.neuron, "seed": 42 if self.kind == "dpa" else 91}
        return result


def test_reported_shared_architecture_is_recorded_without_seed_or_type_order_confusion(tmp_path, paired_inputs):
    _, result, _, _ = run(tmp_path, paired_inputs, ConfiguredDeepPot("dpa"), ConfiguredDeepPot("sog"))
    compared = result["architecture_comparison"]
    assert compared["status"] == "matched_for_checked_fields"
    assert compared["matched_fields"] == ["descriptor.channels", "descriptor.lmax", "descriptor.radial_modes", "descriptor.rcut", "fitting_net.neuron"]
    assert result["models"]["dpa"]["fitting_net_metadata"]["seed"] != result["models"]["sog"]["fitting_net_metadata"]["seed"]
    assert result["fine_tune_parent_not_verified"] is True


@pytest.mark.parametrize("descriptor,neuron,field", [
    ({"channels": 32}, None, "descriptor.channels"),
    ({"lmax": 3}, None, "descriptor.lmax"),
    ({"radial_modes": 2}, None, "descriptor.radial_modes"),
    ({"rcut": 5.0}, None, "descriptor.rcut"),
    ({}, [128, 128], "fitting_net.neuron"),
])
def test_known_architecture_mismatches_are_rejected(tmp_path, paired_inputs, descriptor, neuron, field):
    baseline = ConfiguredDeepPot("dpa")
    sog = ConfiguredDeepPot("sog", descriptor, neuron)
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        run(tmp_path, paired_inputs, baseline, sog)
    assert not (tmp_path / "paired.jsonl").exists()


def test_missing_architecture_fields_are_recorded_as_not_compared(tmp_path, paired_inputs):
    _, result, _, _ = run(tmp_path, paired_inputs, ConfiguredDeepPot("dpa"), FakeDeepPot("sog"))
    assert result["architecture_comparison"]["status"] == "partially_compared"
    assert result["architecture_comparison"]["matched_fields"] == []
    assert len(result["architecture_comparison"]["not_compared_fields"]) == 5


def test_same_checkpoint_and_bad_explicit_condition_rejected(tmp_path, paired_inputs):
    db, dpa, _ = paired_inputs
    with pytest.raises(ValueError, match="charge-spin 0 1"):
        predict(db, dpa, dpa, tmp_path / "bad.jsonl", (0, 0), loader=lambda _: None)
    with pytest.raises(ValueError, match="identical SHA256"):
        predict(db, dpa, dpa, tmp_path / "same.jsonl", (0, 1), loader=lambda _: None)


def test_refuses_noncontiguous_test_ordinal(tmp_path, paired_inputs):
    import sqlite3
    db = paired_inputs[0]
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE records SET ordinal=ordinal+5 WHERE channel='sog_candidates' AND split='test'")
        conn.commit()
    with pytest.raises(ValueError, match="contiguous ordinal"):
        run(tmp_path, paired_inputs)
    assert not (tmp_path / "paired.jsonl").exists()


def test_backend_without_explicit_charge_spin_argument_is_rejected(tmp_path, paired_inputs):
    class NoCondition(FakeDeepPot):
        def eval(self, coords, cells, atom_types, **kwargs):
            raise AssertionError("unsupported backend must never be evaluated")
    with pytest.raises(ValueError, match="does not explicitly support charge_spin"):
        run(tmp_path, paired_inputs, sog_model=NoCondition("sog"))


@pytest.mark.parametrize("key,value", [("charge_multiplicity", [0, 3]), ("spin_multiplicity", 3), ("multiplicity", 2)])
def test_rejects_source_condition_conflicts(tmp_path, paired_inputs, monkeypatch, key, value):
    from cace_sog_training.data import IndexReader
    original = IndexReader.read_atoms
    def marked(self, record):
        atoms = original(self, record)
        atoms.info[key] = value
        return atoms
    monkeypatch.setattr(IndexReader, "read_atoms", marked)
    with pytest.raises(ValueError, match="conflicts"):
        run(tmp_path, paired_inputs)
    assert not (tmp_path / "paired.jsonl").exists()


def test_test_prediction_requires_both_completed_validation_reports(tmp_path, paired_inputs):
    db, dpa, sog = paired_inputs
    with pytest.raises(ValueError, match="validation reports are required"):
        predict(db, dpa, sog, tmp_path / "unvalidated.jsonl", (0, 1), loader=lambda _: None)
    assert not (tmp_path / "unvalidated.jsonl").exists()


@pytest.mark.parametrize("defect", ["status", "kind", "split", "condition", "index", "frames", "frame_hash", "checkpoint", "candidate_hash", "selection"])
def test_mismatched_validation_evidence_never_publishes_test_predictions(tmp_path, paired_inputs, defect):
    from dpa4c_sog.validate import validate
    db, dpa, sog = paired_inputs
    paths = {label: tmp_path / (label + "-val") / "validation.json" for label in ("dpa", "sog")}
    for label, checkpoint in (("dpa", dpa), ("sog", sog)):
        validate(db, label, [checkpoint], paths[label].parent, (0, 1),
                 loader=lambda unused, name=label: FakeDeepPot(name), progress_seconds=1000)
    report = json.loads(paths["sog"].read_text())
    if defect == "status": report["status"] = "partial"
    if defect == "kind": report["kind"] = "dpa"
    if defect == "split": report["split"] = "test"
    if defect == "condition": report["condition"] = [0, 3]
    if defect == "index": report["index_identity"] = "0" * 64
    if defect == "frames": report["frames_per_checkpoint"] += 1
    if defect == "frame_hash": report["frame_identity_sha256"] = "0" * 64
    if defect == "checkpoint": report["selected_checkpoint_sha256"] = "0" * 64
    if defect == "candidate_hash": report["candidates"][0]["checkpoint_sha256"] = "0" * 64
    if defect == "selection": report["selected_index"] = 5
    paths["sog"].write_text(json.dumps(report))
    models = {dpa: FakeDeepPot("dpa"), sog: FakeDeepPot("sog")}
    output = tmp_path / "bad-validation.jsonl"
    with pytest.raises(ValueError):
        predict(db, dpa, sog, output, (0, 1), validation_reports=paths, loader=models.get, progress_seconds=1000)
    assert not output.exists()
    assert json.loads(output.with_name(output.name + ".status.json").read_text())["status"] == "failed"
