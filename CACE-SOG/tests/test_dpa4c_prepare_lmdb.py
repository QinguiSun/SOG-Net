import json
import pytest
from test_dpa4c_convert import make_dataset
from test_dpa4c_index_smoke import release_file
from dpa4c_sog.index_lmdb import convert
from dpa4c_sog.prepare_lmdb import prepare


def setup_data(tmp_path, limit=None):
    pytest.importorskip('lmdb')
    pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path)
    release = tmp_path / 'release.json'
    original = release_file(release)
    data = tmp_path / 'lmdb'
    convert(db, data, map_size=16*1024*1024, limit_per_split=limit)
    return db, release, original, data


def test_production_config_keeps_model_and_train_val_only(tmp_path):
    db, release, original, data = setup_data(tmp_path)
    out = tmp_path / 'config'
    report = prepare(db, release, data, out, 12, 1, 42)
    config = json.loads((out / 'input_dpa4c_lmdb.json').read_text())
    assert config['model'] == original['model']
    assert config['training']['training_data']['systems'] == str(data / 'train.lmdb')
    assert config['training']['training_data']['auto_prob_style'] == 'prob_sys_size'
    assert config['training']['validation_data']['systems'] == str(data / 'val.lmdb')
    assert config['training']['num_steps'] == 12
    assert config['training']['enable_compile'] is False
    assert config['training']['change_bias_after_training'] is False
    assert 'stat_file' not in config['training']
    assert 'test.lmdb' not in json.dumps(config)
    assert report['source_counts'] == {'train': 3, 'val': 2}
    assert report['status'] == 'prepared_only' and report['test_split_read'] is False


def test_reject_debug_and_changed_data(tmp_path):
    db, release, _, data = setup_data(tmp_path, limit=2)
    with pytest.raises(ValueError, match='unsampled'):
        prepare(db, release, data, tmp_path / 'bad-debug', 5, 1, 4)
    data2 = tmp_path / 'complete'
    convert(db, data2, map_size=16*1024*1024)
    with (data2 / 'train.jsonl').open('a') as f:
        f.write('tamper')
    with pytest.raises(ValueError, match='SHA-256'):
        prepare(db, release, data2, tmp_path / 'bad-hash', 5, 1, 4)


def test_checkpoint_schedule_is_configurable_and_recorded(tmp_path):
    db, release, _, data = setup_data(tmp_path)
    out = tmp_path / 'scheduled'
    report = prepare(db, release, data, out, 250, 1, 42,
                     save_every=25, keep_checkpoints=7)
    config = json.loads((out / 'input_dpa4c_lmdb.json').read_text())
    assert config['training']['save_freq'] == report['save_every'] == 25
    assert config['training']['max_ckpt_keep'] == report['keep_checkpoints'] == 7
    # A short requested run must save no later than its final step.
    out2 = tmp_path / 'short'
    report2 = prepare(db, release, data, out2, 8, 1, 42, save_every=25)
    assert report2['save_every'] == 8


@pytest.mark.parametrize('kwargs', [
    {'save_every': 0}, {'save_every': True}, {'keep_checkpoints': 0},
    {'keep_checkpoints': True},
])
def test_checkpoint_schedule_rejects_invalid_values(tmp_path, kwargs):
    with pytest.raises(ValueError):
        prepare(tmp_path / 'missing', tmp_path / 'missing', tmp_path / 'missing',
                tmp_path / 'out', 10, 1, 42, **kwargs)
