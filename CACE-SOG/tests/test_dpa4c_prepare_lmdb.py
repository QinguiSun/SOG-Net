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
    assert config['training']['validation_data']['systems'] == str(data / 'val.lmdb')
    assert config['training']['num_steps'] == 12
    assert config['training']['enable_compile'] is False
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
