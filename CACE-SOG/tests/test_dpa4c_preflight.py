"""Scientific preflight checks against owned, periodic analytic models."""
import json

import numpy as np
import pytest

from cace_sog_training.data import IndexReader, build_index
from dpa4c_sog.preflight import preflight
from elementa_fixtures import exported_fixture
from test_dpa4c_predict import FakeDeepPot


class PeriodicModel(FakeDeepPot):
    """E = sum(1 - cos(2π fractional coordinate)), F = -dE/dr."""
    def __init__(self, kind="dpa", bad_force=False, break_periodicity=False):
        super().__init__(kind)
        self.bad_force = bad_force
        self.break_periodicity = break_periodicity

    def eval(self, coords, cells, atom_types, atomic=False, charge_spin=None):
        assert np.array_equal(charge_spin, [[0., 1.]])
        assert [self.types[int(i)] for i in atom_types] == ["Na", "Cl"]
        coords = np.asarray(coords, dtype=float).reshape(-1, 3)
        cell = np.asarray(cells, dtype=float).reshape(3, 3)
        if self.break_periodicity:
            energy = np.sum(coords**2)
            forces = -2.0 * coords
        else:
            inv = np.linalg.inv(cell)
            frac = coords @ inv
            energy = np.sum(1.0 - np.cos(2.0 * np.pi * frac))
            forces = -(2.0 * np.pi * np.sin(2.0 * np.pi * frac)) @ inv.T
        if self.bad_force:
            forces[0, 0] += 1.0
        return np.array([[energy]]), forces[None], np.zeros((1, 9))


@pytest.fixture
def inputs(tmp_path):
    exported_fixture(tmp_path / "export", channels=("sog_candidates",), train=3, val=1, test=1)
    database = tmp_path / "index.sqlite3"
    build_index(tmp_path / "export", database, channels=["sog_candidates"])
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"owned fake checkpoint, not a DeepMD model")
    return database, checkpoint


def test_central_forces_and_lattice_translation_only_read_train(inputs, tmp_path, monkeypatch):
    database, checkpoint = inputs
    original = IndexReader.iter_records
    observed = []
    def only_train(self, channels, split):
        observed.append(split)
        assert tuple(channels) == ("sog_candidates",) and split == "train"
        yield from original(self, channels, split)
    monkeypatch.setattr(IndexReader, "iter_records", only_train)
    path = tmp_path / "report.json"
    result = preflight(database, checkpoint, "dpa", path, [0, 1],
        frames=2, atoms_per_frame=2, force_atol=1e-5, force_rtol=0,
        energy_atol=1e-8, periodic_force_atol=1e-8, loader=lambda _: PeriodicModel())
    assert result["status"] == "passed" and observed == ["train"]
    assert result["frames_checked"] == 2 and result["force_components_checked"] == 12
    assert result["maximum_force_fd_abs_difference_eV_A"] < 1e-5
    assert len(result["frames"][0]["frame_sha256"]) == 64
    assert result["test_split_read"] is False
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "passed"


def test_wrong_force_is_failed_and_diagnostics_are_persisted(inputs, tmp_path):
    database, checkpoint = inputs
    path = tmp_path / "bad-force.json"
    with pytest.raises(ValueError, match="checks exceeded"):
        preflight(database, checkpoint, "dpa", path, [0, 1], frames=1,
            force_atol=0.01, loader=lambda _: PeriodicModel(bad_force=True))
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert any("force finite difference" in item for item in result["failures"])
    assert result["maximum_force_fd_abs_difference_eV_A"] > 0.9


def test_nonperiodic_model_fails_translation_even_with_correct_forces(inputs, tmp_path):
    database, checkpoint = inputs
    path = tmp_path / "bad-periodic.json"
    with pytest.raises(ValueError, match="checks exceeded"):
        preflight(database, checkpoint, "dpa", path, [0, 1], frames=1,
            force_atol=1e-5, force_rtol=0, loader=lambda _: PeriodicModel(break_periodicity=True))
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["frames"][0]["translation"]["passed"] is False
    assert result["maximum_force_fd_abs_difference_eV_A"] < 1e-5


def test_sog_kind_and_bounded_sample(inputs, tmp_path):
    database, checkpoint = inputs
    result = preflight(database, checkpoint, "sog", tmp_path / "sog.json", [0, 1],
        frames=8, atoms_per_frame=8, loader=lambda _: PeriodicModel(kind="sog"))
    assert result["status"] == "passed" and result["frames_checked"] == 3
    assert result["force_components_checked"] == 18
    assert result["model"]["long_range"]["kernel"]["type"] == "sog"


@pytest.mark.parametrize("kwargs", [{"frames": 0}, {"frames": 257}, {"atoms_per_frame": 65},
    {"step": 0}, {"step": float("nan")}, {"force_rtol": -1}, {"force_atol": 0}])
def test_invalid_settings_do_not_create_report(inputs, tmp_path, kwargs):
    database, checkpoint = inputs
    path = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        preflight(database, checkpoint, "dpa", path, [0, 1], loader=lambda _: PeriodicModel(), **kwargs)
    assert not path.exists()


def test_existing_report_or_input_path_never_overwritten(inputs, tmp_path):
    database, checkpoint = inputs
    output = tmp_path / "exists.json"
    output.write_text("existing")
    with pytest.raises(ValueError, match="NEW report"):
        preflight(database, checkpoint, "dpa", output, [0, 1], loader=lambda _: PeriodicModel())
    assert output.read_text() == "existing"
    with pytest.raises(ValueError, match="NEW report"):
        preflight(database, checkpoint, "dpa", checkpoint, [0, 1], loader=lambda _: PeriodicModel())
    with pytest.raises(ValueError, match="outside the source"):
        preflight(database, checkpoint, "dpa", tmp_path / "export/report.json", [0, 1],
                  loader=lambda _: PeriodicModel())


def test_model_load_failure_writes_failed_report(inputs, tmp_path):
    database, checkpoint = inputs
    output = tmp_path / "failed.json"
    def fail(_):
        raise RuntimeError("no real DeepMD backend in this test")
    with pytest.raises(RuntimeError, match="no real DeepMD"):
        preflight(database, checkpoint, "dpa", output, [0, 1], loader=fail)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_neighboring_temp_file_is_not_overwritten(inputs, tmp_path):
    database, checkpoint = inputs
    output = tmp_path / "report.json"
    neighbor = tmp_path / "report.json.tmp"
    neighbor.write_text("owned file", encoding="utf-8")
    result = preflight(database, checkpoint, "dpa", output, [0, 1],
                       frames=1, loader=lambda _: PeriodicModel())
    assert result["status"] == "passed"
    assert neighbor.read_text(encoding="utf-8") == "owned file"
