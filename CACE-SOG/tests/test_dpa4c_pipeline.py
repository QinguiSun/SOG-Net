"""Small offline Windows-to-SCNet workflow; all DeepPot models are fakes."""
import json
import os
from pathlib import Path
import shutil

from ase.data import chemical_symbols

from cace_sog_training.data import build_index
from cace_sog_training.transfer import seal_index, bind_index
from dpa4c_sog import integrity
from dpa4c_sog.alignment import verify_alignment
from dpa4c_sog.convert import convert
from dpa4c_sog.metrics import evaluate_predictions
from dpa4c_sog.predict import predict
from dpa4c_sog.prepare import prepare
from dpa4c_sog.validate import validate
from elementa_fixtures import exported_fixture
from test_dpa4c_predict import FakeDeepPot
from test_dpa4c_validate import ScaleModel


def test_complete_cross_machine_pipeline_with_mocked_predictions(tmp_path):
    local = tmp_path / "Windows local"
    remote = tmp_path / "SCNet copy"
    local.mkdir()
    remote.mkdir()
    source = local / "export"
    exported_fixture(source, channels=("sog_candidates",), train=3, val=2, test=2)
    index = local / "training.sqlite3"
    build_index(source, index, channels=["sog_candidates"])
    converted = local / "converted"
    conversion = convert(index, converted, (0, 1), set_frames=2)
    assert conversion["status"] == "complete"
    assert conversion["counts"] == {"train": 3, "val": 2, "test": 2}

    manifest = local / "converted.manifest.jsonl"
    integrity_result = integrity.seal(converted, manifest)
    bundle = local / "bundle"
    source_seal = seal_index(index, bundle, progress=False)
    assert source_seal["indexed_frames"] == 7

    # This is a local simulation of binary upload and different target mtimes.
    remote_export = remote / "export"
    remote_converted = remote / "converted"
    remote_bundle = remote / "bundle"
    remote_manifest = remote / "converted.manifest.jsonl"
    shutil.copytree(source, remote_export)
    shutil.copytree(converted, remote_converted)
    shutil.copytree(bundle, remote_bundle)
    shutil.copy2(manifest, remote_manifest)
    for path in remote_export.rglob("*"):
        if path.is_file():
            os.utime(path, (1700000000, 1700000000))
    bound_index = remote / "bound.sqlite3"
    assert bind_index(remote_bundle, remote_export, bound_index, progress=False)["status"] == "complete"
    integrity_report = remote / "verified.json"
    verification = integrity.verify(remote_converted, remote_manifest, integrity_result["manifest_sha256"],
                                    report=integrity_report)
    assert verification["status"] == "complete"
    alignment_path = remote / "alignment.json"
    alignment = verify_alignment(bound_index, remote_converted, alignment_path)
    assert alignment["counts"] == {"train": 3, "val": 2, "test": 2}
    assert alignment["frames_checked"] == 7 and alignment["metadata_identities_differ"]

    release = remote / "fake_release.json"
    release.write_text(json.dumps({"model": {"type_map": chemical_symbols[1:119],
        "descriptor": {"type": "dpa4c", "add_chg_spin_ebd": True, "default_chg_spin": [0, 1]},
        "fitting_net": {"neuron": [8, 8]}}, "training": {},
        "learning_rate": {"start_lr": 0.001, "stop_lr": 1e-6}}))
    inputs = remote / "inputs"
    prep = prepare(release, remote_converted, inputs, train_steps=4, sog_steps=3,
                   integrity_report=integrity_report, alignment_report=alignment_path)
    assert prep["no_training_executed"] and prep["bound_index_identity"] == alignment["bound_index_identity"]
    baseline_cfg = json.loads((inputs / "input_dpa4c.json").read_text())
    sog_cfg = json.loads((inputs / "input_sog_experimental.json").read_text())
    assert baseline_cfg["validating"]["full_validation"] is False
    assert sog_cfg["model"]["long_range"]["kernel"]["type"] == "sog"

    # Fake checkpoints and model classes establish pipeline contracts only;
    # they do not demonstrate DeepMD training, inference, CUDA or accuracy.
    dpa_paths = [remote / "dpa-a.pt", remote / "dpa-b.pt"]
    sog_paths = [remote / "sog-a.pt", remote / "sog-b.pt"]
    for path in dpa_paths + sog_paths:
        path.write_bytes(("mock checkpoint: " + path.name).encode())
    fake_models = {dpa_paths[0]: ScaleModel("dpa", 1), dpa_paths[1]: ScaleModel("dpa", 0),
                   sog_paths[0]: ScaleModel("sog", 1), sog_paths[1]: ScaleModel("sog", 0)}
    base_val = validate(bound_index, "dpa", dpa_paths, remote / "dpa-val", (0, 1), loader=fake_models.get)
    sog_val = validate(bound_index, "sog", sog_paths, remote / "sog-val", (0, 1), loader=fake_models.get)
    assert base_val["selected_index"] == sog_val["selected_index"] == 1
    assert base_val["frames_per_checkpoint"] == sog_val["frames_per_checkpoint"] == 2
    assert base_val["frame_identity_sha256"] == sog_val["frame_identity_sha256"]
    paired = remote / "paired.jsonl"
    result = predict(bound_index, dpa_paths[1], sog_paths[1], paired, (0, 1), loader=fake_models.get)
    assert result["status"] == "complete" and result["frames_written"] == 2
    assert result["index_identity"] == alignment["bound_index_identity"]
    report = evaluate_predictions(paired, remote / "comparison", bins=16)
    assert report["status"] == "complete" and report["paired_frames"] == 2
    assert report["metrics"]["dpa"]["energy"]["count"] == 2
    assert report["metrics"]["sog"]["force"]["count"] == 12
    assert report["prediction_provenance"]["status"]["index_identity"] == alignment["bound_index_identity"]
    assert report["prediction_provenance"]["prediction_sha256_and_frame_counts_verified"]
    assert (remote / "comparison/dpa_parity.png").is_file()
    assert (remote / "comparison/sog_parity.png").is_file()
