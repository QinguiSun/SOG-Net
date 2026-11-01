import json
import hashlib

import pytest

from cace_sog_training.data import build_index
from dpa4c_sog.baseline_predict import predict
from dpa4c_sog.baseline_metrics import evaluate_predictions
from dpa4c_sog.baseline_merge import merge
from dpa4c_sog.validate import validate
from elementa_fixtures import exported_fixture
from test_dpa4c_predict import FakeDeepPot


class Shifted(FakeDeepPot):
    def eval(self, coords, cells, atom_types, atomic=False, charge_spin=None):
        energy, forces, virial = super().eval(coords, cells, atom_types, atomic, charge_spin)
        return energy + 4, forces + 2, virial


def test_original_and_finetuned_paired_test_and_plots(tmp_path):
    source = tmp_path / 'export'
    exported_fixture(source, train=2, val=1, test=2)
    db = tmp_path / 'index.sqlite3'
    build_index(source, db)
    before, after = tmp_path / 'original.pt', tmp_path / 'fine.pt'
    before.write_bytes(b'raw-pretrained')
    after.write_bytes(b'fine-tuned')
    val_dir = tmp_path / 'val'
    validate(db, 'dpa', [after], val_dir, (0, 1),
             loader=lambda _: FakeDeepPot('dpa'), progress_seconds=1000)
    predictions = tmp_path / 'test.jsonl'
    result = predict(db, before, after, predictions, (0, 1),
                     validation_report=val_dir / 'validation.json',
                     loader=lambda p: Shifted('dpa') if p == after else FakeDeepPot('dpa'), progress_seconds=1000,
                     expected_pretrained_sha256=hashlib.sha256(b'raw-pretrained').hexdigest())
    assert result['status'] == 'complete' and result['frames_written'] == 2
    assert set(result['checkpoints']) == {'pretrained', 'finetuned'}
    rows = [json.loads(s) for s in predictions.read_text().splitlines()]
    assert len(rows) == 2 and all(x['split'] == 'test' for x in rows)
    assert 'pretrained_energy_eV' in rows[0] and 'finetuned_energy_eV' in rows[0]
    report = evaluate_predictions(predictions, tmp_path / 'plots', bins=16)
    assert report['paired_frames'] == 2
    assert report['metrics']['pretrained']['energy']['count'] == 2
    assert report['metrics']['finetuned']['force']['count'] == 12
    assert report['metrics']['finetuned']['energy']['mean_difference'] == pytest.approx(
        report['metrics']['pretrained']['energy']['mean_difference'] + 2)
    assert report['metrics']['finetuned']['force']['mean_difference'] == pytest.approx(
        report['metrics']['pretrained']['force']['mean_difference'] + 2)
    assert (tmp_path / 'plots/pretrained_parity.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path / 'plots/finetuned_parity.png').read_bytes().startswith(b'\x89PNG')
    with pytest.raises(ValueError, match='already exists'):
        predict(db, before, after, predictions, (0, 1), validation_report=val_dir / 'validation.json')


def test_shards_require_full_nonoverlapping_coverage(tmp_path, monkeypatch):
    source = tmp_path / 'export'
    exported_fixture(source, train=2, val=1, test=2)
    db = tmp_path / 'index.sqlite3'
    build_index(source, db)
    before, after = tmp_path / 'original.pt', tmp_path / 'fine.pt'
    before.write_bytes(b'raw-pretrained')
    after.write_bytes(b'fine-tuned')
    val_dir = tmp_path / 'val'
    validate(db, 'dpa', [after], val_dir, (0, 1),
             loader=lambda _: FakeDeepPot('dpa'), progress_seconds=1000)
    shards = [tmp_path / 'part0.jsonl', tmp_path / 'part1.jsonl']
    for i, path in enumerate(shards):
        state = predict(db, before, after, path, (0, 1), start=i, end=i+1,
                        validation_report=val_dir / 'validation.json',
                        loader=lambda _: FakeDeepPot('dpa'), progress_seconds=1000,
                        expected_pretrained_sha256=hashlib.sha256(b'raw-pretrained').hexdigest())
        assert state['is_shard'] and state['expected_frames'] == 1
    with pytest.raises(ValueError, match='individual shard'):
        evaluate_predictions(shards[0], tmp_path / 'bad-plot', bins=16)
    with pytest.raises(ValueError, match='entire test'):
        merge(shards[:1], tmp_path / 'incomplete.jsonl')
    with pytest.raises(ValueError, match='overlap'):
        merge([shards[0], shards[0], shards[1]], tmp_path / 'overlap.jsonl')
    complete = tmp_path / 'complete.jsonl'
    status = merge(shards[::-1], complete)
    assert status['status'] == 'complete' and status['frames_written'] == 2
    assert not status['is_shard']
    rows = [json.loads(line) for line in complete.read_text().splitlines()]
    assert [r['frame_key'] for r in rows] == ['sog_candidates/test/0', 'sog_candidates/test/1']
    import dpa4c_sog.baseline_metrics as m
    monkeypatch.setattr(m, '_render', lambda *args: None)
    assert m.evaluate_predictions(complete, tmp_path / 'report', bins=16)['paired_frames'] == 2
    with shards[1].open('ab') as f:
        f.write(b'{}\n')
    with pytest.raises(ValueError):
        merge(shards, tmp_path / 'tampered.jsonl')
    assert not (tmp_path / 'tampered.jsonl').exists()


def test_calibrated_checkpoint_rejected_as_original(tmp_path):
    source = tmp_path / 'export'
    exported_fixture(source, train=2, val=1, test=2)
    db = tmp_path / 'index.sqlite3'
    build_index(source, db)
    calibrated = tmp_path / 'calibrated.pt'
    trained = tmp_path / 'trained.pt'
    calibrated.write_bytes(b'train-only-calibrated')
    trained.write_bytes(b'fine-tuned')
    output = tmp_path / 'wrong-baseline.jsonl'
    with pytest.raises(ValueError, match='original released DPA4C checkpoint'):
        predict(db, calibrated, trained, output, (0, 1),
                validation_report=tmp_path / 'unused.json',
                loader=lambda _: FakeDeepPot('dpa'))
    assert not output.exists()
    assert json.loads(output.with_name(output.name + '.status.json').read_text())['status'] == 'failed'
