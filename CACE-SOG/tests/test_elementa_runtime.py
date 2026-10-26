import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cace_sog_training.data import DEFAULT_CHANNELS, build_index
from cace_sog_training import runtime
from elementa_fixtures import exported_fixture


def arguments(tmp_path, database):
    return SimpleNamespace(database=database, channels=list(DEFAULT_CHANNELS),
        spin_policy="exclude", spin_policy_note=None, run_dir=tmp_path / "run",
        model_kind="sog", device="cpu", dtype="float64", cutoff=3., n_dl=2., max_modes=250000,
        atom_basis=2, radial_basis=2, max_l=1, max_nu=2, message_passing=0, seed=17,
        batch_size=4, workers=0, lr=.001, energy_weight=1., force_weight=10.,
        max_grad_norm=10., checkpoint_every=2, validate_every_steps=0, shuffle="affine",
        progress_seconds=10000., epochs=1, resume=None, max_steps=None)


@pytest.fixture
def database(tmp_path):
    exported_fixture(tmp_path / "export")
    db = tmp_path / "index.sqlite3"
    build_index(tmp_path / "export", db, commit_every=3)
    return db


def test_sampler_is_constant_state_exact_permutation_with_restart():
    for size in (1, 8, 101, 1024):
        full = list(runtime.EpochSampler(size, 42, 3))
        assert sorted(full) == list(range(size))
        start = size // 2
        assert list(runtime.EpochSampler(size, 42, 3, start)) == full[start:]


def test_streaming_metrics_do_not_retain_predictions():
    metrics = runtime.StreamingMetrics()
    for _ in range(5000):
        metrics.update("all", torch.tensor([.1, -.1]), torch.tensor([.2, -.2, .2]))
    result = metrics.result()["all"]
    assert result["frames"] == 10000
    assert result["energy_MAE_eV_atom"] == pytest.approx(.1)
    assert result["force_component_RMSE_eV_A"] == pytest.approx(.2)
    assert len(json.dumps(metrics.state)) < 400


def test_train_restart_matches_uninterrupted_and_selects_val_best(database, tmp_path):
    full = arguments(tmp_path, database)
    complete = runtime.run_train(full)
    assert complete["status"] == "complete"
    assert complete["train_frames_per_epoch"] == 15
    expected = runtime.load_checkpoint(full.run_dir / "last.pt")
    partial = arguments(tmp_path, database)
    partial.run_dir = tmp_path / "restart"
    partial.max_steps = 2
    interrupted = runtime.run_train(partial)
    assert interrupted["status"] == "partial" and interrupted["offset"] == 8
    partial.resume = partial.run_dir / "last.pt"
    partial.max_steps = None
    assert runtime.run_train(partial)["status"] == "complete"
    actual = runtime.load_checkpoint(partial.run_dir / "last.pt")
    assert actual["epoch"] == 1 and actual["offset"] == 0 and actual["step"] == 4
    for key, value in expected["model_state"].items():
        torch.testing.assert_close(actual["model_state"][key], value, rtol=0, atol=0)
    epochs = [json.loads(line) for line in (partial.run_dir / "epochs.jsonl").read_text().splitlines()]
    assert len(epochs) == 1 and epochs[0]["train"]["all"]["frames"] == 15
    assert epochs[0]["val"]["all"]["frames"] == 6
    evaluation = SimpleNamespace(database=database, checkpoint=partial.run_dir / "best.pt", channels=None,
        split="test", device="cpu", batch_size=4, workers=0, progress_seconds=10000., output=tmp_path / "test")
    report = runtime.run_evaluate(evaluation)
    assert report["metrics"]["all"]["frames"] == 6
    assert set(report["metrics"]) == {"all", *DEFAULT_CHANNELS}
    assert all(report["metrics"][channel]["frames"] == 2 for channel in DEFAULT_CHANNELS)


def test_baseline_is_train_only_with_explicit_rank_deficiency(database, tmp_path):
    with runtime.IndexReader(database) as reader:
        e0, stats = runtime.train_baseline(reader, DEFAULT_CHANNELS)
    assert stats["frames"] == 15
    assert set(e0) == {11, 17}
    assert stats["normal_matrix_rank"] == 1


def test_unknown_validation_elements_fail_before_optimization(tmp_path):
    exported_fixture(tmp_path / "export", unknown_val=True)
    db = tmp_path / "index.sqlite3"
    build_index(tmp_path / "export", db)
    args = arguments(tmp_path, db)
    with pytest.raises(ValueError, match="val has elements absent"):
        runtime.run_train(args)
    assert not (args.run_dir / "last.pt").exists()


def test_resume_changed_scope_is_rejected(database, tmp_path):
    args = arguments(tmp_path, database)
    args.max_steps = 1
    runtime.run_train(args)
    args.resume = args.run_dir / "last.pt"
    args.batch_size = 2
    with pytest.raises(ValueError, match="settings differ"):
        runtime.run_train(args)


def test_restart_at_debug_cap_does_not_perform_an_extra_update(database, tmp_path):
    args = arguments(tmp_path, database)
    args.max_steps = 1
    runtime.run_train(args)
    checkpoint = args.run_dir / "last.pt"
    before = checkpoint.read_bytes()
    args.resume = checkpoint
    report = runtime.run_train(args)
    assert report["status"] == "partial" and report["step"] == 1
    assert checkpoint.read_bytes() == before
    args.max_steps = None
    args.n_dl = 1.
    with pytest.raises(ValueError, match="architecture/grid settings differ"):
        runtime.run_train(args)


def test_spin_is_opt_in_and_pending_is_not_a_training_channel():
    with pytest.raises(ValueError, match="spin_review requires"):
        runtime.validate_scope(["spin_review"])
    with pytest.raises(ValueError, match="spin_review requires"):
        runtime.validate_scope(["spin_review"], "consistent-state")
    assert runtime.validate_scope(["spin_review"], "consistent-state", "reviewed fixed AFM branch") == ("spin_review",)
    with pytest.raises(ValueError, match="approved channels"):
        runtime.validate_scope(["pending_review"])


def test_smoke_and_grid_convergence_are_actual_model_checks(database, tmp_path):
    args = arguments(tmp_path, database)
    args.output = tmp_path / "smoke"
    report = runtime.run_smoke(args)
    assert report["updated_parameter_count"] > 0
    assert max(report["reload_max_abs_errors"].values()) < 1e-9
    args.output = tmp_path / "grid"
    args.n_dl, args.samples, args.checkpoint = [3., 2.], 2, None
    comparison = runtime.run_converge(args)
    assert comparison["samples"] == 2 and len(comparison["grids"]) == 2
    finest = next(row for row in comparison["grids"] if row["n_dl"] == 2.)
    assert finest["force_max_abs_eV_A_vs_finest"] == 0


def test_single_writer_lock():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with runtime.run_lock(tmp):
            with pytest.raises(ValueError, match="another training"):
                with runtime.run_lock(tmp):
                    pass
