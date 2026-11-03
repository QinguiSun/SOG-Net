import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
from test_dpa4c_convert import make_dataset
from test_dpa4c_index_smoke import release_file
from dpa4c_sog.index_lmdb import convert
from dpa4c_sog.prepare_lmdb import prepare
from dpa4c_sog import train_run_guard as guard


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    pytest.importorskip('lmdb'); pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path)
    release = tmp_path / 'release.json'; release_file(release)
    pretrained = tmp_path / 'pretrained.pt'; pretrained.write_bytes(b'original')
    monkeypatch.setattr(guard, 'PRETRAINED_SHA256', digest(pretrained))
    monkeypatch.setattr(guard, 'RELEASE_SHA256', digest(release))
    data = tmp_path / 'data'; convert(db, data, map_size=16*1024*1024)
    prepared = tmp_path / 'prepared'; prepare(db, release, data, prepared, 10, 1, 42)
    return db, release, pretrained, data, prepared


def test_start_restart_and_tamper_guards(tmp_path, monkeypatch):
    db, release, pretrained, data, prepared = fixture(tmp_path, monkeypatch)
    commit = 'a'*40
    checked = guard.checked_files(prepared, db, release, pretrained, commit)
    run = tmp_path / 'run'; run.mkdir(); guard.start(run, checked)
    with pytest.raises(FileExistsError): guard.start(run, checked)
    first = run / 'model.ckpt-5.pt'; first.write_bytes(b'first')
    second = run / 'model.ckpt-10.pt'; second.write_bytes(b'latest')
    with pytest.raises(ValueError, match='latest'):
        guard._validate_resume_provenance(run, checked, first, digest(first))
    with pytest.raises(ValueError, match='SHA'):
        guard._validate_resume_provenance(run, checked, second, '0'*64)
    guard._validate_resume_provenance(run, checked, second, digest(second))
    with pytest.raises(RuntimeError, match='sampler replay'):
        guard.resume(run, checked, second, digest(second))
    with pytest.raises(ValueError, match='provenance'):
        guard._validate_resume_provenance(run, dict(checked, repository_commit='b'*40), second, digest(second))
    with (run / 'input_dpa4c_lmdb.json').open('a') as f: f.write(' ')
    with pytest.raises(ValueError, match='configuration'):
        guard._validate_resume_provenance(run, checked, second, digest(second))
    with (data / 'train.jsonl').open('a') as f: f.write('tamper')
    with pytest.raises(ValueError, match='hash mismatch'):
        guard.checked_files(prepared, db, release, pretrained, commit)


def test_debug_and_wrong_weights_rejected(tmp_path, monkeypatch):
    db, release, pretrained, data, prepared = fixture(tmp_path, monkeypatch)
    pretrained.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint SHA'):
        guard.checked_files(prepared, db, release, pretrained, 'a'*40)
    pretrained.write_bytes(b'original')
    path = prepared / 'preparation.json'
    report = json.loads(path.read_text()); report['debug_only'] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='non-debug'):
        guard.checked_files(prepared, db, release, pretrained, 'a'*40)


def test_batch_script_syntax():
    script = Path(__file__).resolve().parent.parent / 'examples/elementa/dpa4c_train_lmdb.sbatch'
    subprocess.run(['bash', '-n', str(script)], check=True)


def test_calibrated_initialization_is_bound_to_index_and_checkpoint(tmp_path, monkeypatch):
    db, release, pretrained, data, prepared = fixture(tmp_path, monkeypatch)
    commit = 'a' * 40
    raw = guard.checked_files(prepared, db, release, pretrained, commit)
    calibrated = tmp_path / 'calibrated.pt'; calibrated.write_bytes(b'pilot-bias')
    monkeypatch.setattr(guard, 'PILOT_BIAS_SHA256', digest(calibrated))
    monkeypatch.setattr(guard, 'PILOT_INDEX_IDENTITY', raw['source_index_identity'])
    checked = guard.checked_files(prepared, db, release, pretrained, commit,
        initialization='pilot-calibrated', calibrated=calibrated)
    assert checked['initialization'] == 'pilot-calibrated'
    assert checked['pretrained_sha256'] == digest(pretrained)
    assert checked['calibrated_checkpoint_sha256'] == digest(calibrated)
    run = tmp_path / 'run'; run.mkdir(); guard.start(run, checked)
    checkpoint = run / 'model.ckpt-10.pt'; checkpoint.write_bytes(b'weights')
    with pytest.raises(ValueError, match='provenance'):
        guard._validate_resume_provenance(run, raw, checkpoint, digest(checkpoint))
    guard._validate_resume_provenance(run, checked, checkpoint, digest(checkpoint))
    calibrated.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='calibrated checkpoint SHA'):
        guard.checked_files(prepared, db, release, pretrained, commit,
            initialization='pilot-calibrated', calibrated=calibrated)
    calibrated.write_bytes(b'pilot-bias')
    monkeypatch.setattr(guard, 'PILOT_INDEX_IDENTITY', '0' * 64)
    with pytest.raises(ValueError, match='source index identity'):
        guard.checked_files(prepared, db, release, pretrained, commit,
            initialization='pilot-calibrated', calibrated=calibrated)


def test_restart_rejects_checkpoint_schedule_changes(tmp_path, monkeypatch):
    db, release, pretrained, data, _ = fixture(tmp_path, monkeypatch)
    prepared = tmp_path / 'prepared-scheduled'
    prepare(db, release, data, prepared, 250, 1, 42, save_every=25, keep_checkpoints=7)
    checked = guard.checked_files(prepared, db, release, pretrained, 'a' * 40)
    run = tmp_path / 'run-scheduled'; run.mkdir(); guard.start(run, checked)
    checkpoint = run / 'model.ckpt-25.pt'; checkpoint.write_bytes(b'weights')
    guard._validate_resume_provenance(run, checked, checkpoint, digest(checkpoint))
    config_path = run / 'input_dpa4c_lmdb.json'
    config = json.loads(config_path.read_text())
    assert config['training']['save_freq'] == 25
    config['training']['save_freq'] = 50
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='configuration'):
        guard._validate_resume_provenance(run, checked, checkpoint, digest(checkpoint))


def test_resume_cli_rejects_before_accessing_inputs(tmp_path):
    command = [
        sys.executable, '-m', 'dpa4c_sog.train_run_guard', '--mode', 'resume',
        '--prepared', str(tmp_path / 'missing'), '--database', str(tmp_path / 'missing'),
        '--release', str(tmp_path / 'missing'), '--pretrained', str(tmp_path / 'missing'),
        '--repo-commit', 'a' * 40, '--run', str(tmp_path / 'missing'),
        '--initialization', 'raw-finetune',
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'sampler replay' in result.stderr
