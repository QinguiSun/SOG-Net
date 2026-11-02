import hashlib
import json
from pathlib import Path

import pytest
from dpa4c_sog.sampler_recovery import DurableSamplerResume, inspect_pairs


FINGERPRINT = 'a' * 64


def make_pair(root, step):
    cp = root / f'model.ckpt-{step}.pt'
    cp.write_bytes(f'checkpoint {step}'.encode())
    metadata = {'version': 1, 'step': step, 'fingerprint': FINGERPRINT,
                'checkpoint_sha256': hashlib.sha256(cp.read_bytes()).hexdigest(),
                'batch_counts': {'train': step, 'val': step},
                'seeds': {'train': 42, 'val': 42}}
    Path(str(cp) + '.sampler.json').write_text(json.dumps(metadata))
    return cp


def test_selects_latest_complete_and_reports_orphans_without_mutating(tmp_path):
    make_pair(tmp_path, 1)
    make_pair(tmp_path, 2)
    orphan = tmp_path / 'model.ckpt-3.pt'
    orphan.write_bytes(b'incomplete')
    result = inspect_pairs(tmp_path, FINGERPRINT)
    assert result['selected']['step'] == 2
    assert [x['step'] for x in result['conflicts']] == [3]
    assert not result['safe_to_launch'] and orphan.read_bytes() == b'incomplete'


def test_rejects_mismatch_corruption_and_symlinks(tmp_path):
    cp = make_pair(tmp_path, 5)
    assert inspect_pairs(tmp_path, FINGERPRINT)['safe_to_launch']
    cp.write_bytes(b'changed')
    assert inspect_pairs(tmp_path, FINGERPRINT)['selected'] is None
    assert inspect_pairs(tmp_path, 'b' * 64)['selected'] is None
    cp.unlink()
    cp.symlink_to(tmp_path / 'elsewhere')
    assert not inspect_pairs(tmp_path, FINGERPRINT)['safe_to_launch']
    with pytest.raises(ValueError): inspect_pairs(tmp_path, 'bad')


def test_durable_publication_and_no_overwrite(tmp_path):
    cp = tmp_path / 'model.ckpt-1.pt'; cp.write_bytes(b'model')
    state = DurableSamplerResume(tmp_path/'train', tmp_path/'val', FINGERPRINT)
    state.seeds.update(train=42, val=42)
    state.counts.update(train=1, val=1)
    state.checkpoint(cp, 1)
    assert inspect_pairs(tmp_path, FINGERPRINT)['safe_to_launch']
    with pytest.raises(FileExistsError): state.checkpoint(cp, 1)
    with pytest.raises(ValueError): state.checkpoint(cp, 2)


def test_missing_sidecar_and_malformed_metadata(tmp_path):
    cp = make_pair(tmp_path, 7)
    side = Path(str(cp)+'.sampler.json')
    side.write_text('[]')
    assert not inspect_pairs(tmp_path, FINGERPRINT)['safe_to_launch']
    side.unlink()
    assert not inspect_pairs(tmp_path, FINGERPRINT)['safe_to_launch']


def test_rotated_old_checkpoint_reports_stray_sidecar(tmp_path):
    old = make_pair(tmp_path, 1)
    make_pair(tmp_path, 2)
    old.unlink()  # A checkpoint rotation may leave its sidecar behind.
    result = inspect_pairs(tmp_path, FINGERPRINT)
    assert result['selected']['step'] == 2
    assert result['invalid'][0]['step'] == 1
    assert result['safe_to_launch'] and not result['conflicts']


def test_sidecar_wrong_step_or_seed_rejects_checkpoint(tmp_path):
    cp = make_pair(tmp_path, 9)
    side = Path(str(cp) + '.sampler.json')
    data = json.loads(side.read_text())
    data['step'] = 8
    side.write_text(json.dumps(data))
    assert inspect_pairs(tmp_path, FINGERPRINT)['selected'] is None
    data['step'] = 9
    data['seeds']['train'] = None
    side.write_text(json.dumps(data))
    assert inspect_pairs(tmp_path, FINGERPRINT)['selected'] is None
