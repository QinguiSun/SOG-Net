"""Batch-eight restrictions and sampler-cursor semantics, without GPU claims."""
import copy
import json
from types import SimpleNamespace

import pytest

from dpa4c_sog.restricted_train_batch8_experimental import fingerprint, validate_config
from dpa4c_sog.sampler_resume_experimental import SamplerResume


def config(tmp_path):
    train, val = tmp_path / 'train.lmdb', tmp_path / 'val.lmdb'
    cfg = {'model': {'descriptor': {'type': 'dpa4c'}}, 'training': {
        'training_data': {'systems': str(train), 'batch_size': 8, 'auto_prob_style': 'prob_sys_size'},
        'validation_data': {'systems': str(val), 'batch_size': 1, 'numb_batch': 1},
        'enable_compile': False, 'enable_ema': False, 'seed': 42,
        'num_steps': 10, 'save_freq': 2, 'max_ckpt_keep': 3}}
    return cfg, train, val


def test_fixed_train_eight_and_val_one(tmp_path):
    cfg, train, val = config(tmp_path)
    validate_config(cfg, train, val)
    for key, value in [('training_data', 1), ('training_data', 4), ('validation_data', 8)]:
        bad = copy.deepcopy(cfg)
        bad['training'][key]['batch_size'] = value
        with pytest.raises(ValueError):
            validate_config(bad, train, val)
    assert fingerprint({'batch_size': 8}) != fingerprint({'batch_size': 1})


class Iterator:
    def __init__(self, batches):
        self.batches = batches
        self.index = 0
        self._pending = None
        self._deferred_indices = None

    def _next_indices(self):
        result = self.batches[self.index % len(self.batches)]
        self.index += 1
        return result[:]

    def take(self):
        result = self._next_indices() if self._deferred_indices is None else self._deferred_indices
        self._deferred_indices = self._next_indices()
        return result


class Data:
    def __init__(self, path, batches):
        self._reader = SimpleNamespace(lmdb_path=str(path))
        self._sampler = SimpleNamespace(_seed=42)
        self._batch_iterator = Iterator(batches)


def take(data):
    return data._batch_iterator.take()


def test_resume_counts_batches_not_frames_with_tail_and_epoch_boundary(tmp_path):
    train, val = tmp_path / 'train', tmp_path / 'val'
    batches = [list(range(8)), [8, 9], [10], list(range(11, 19))]
    validation = [[20], [21], [22]]
    state = SamplerResume(train, val, 'batch8-identity')
    td, vd = Data(train, batches), Data(val, validation)
    observed = [state.wrap_batch(take, td) for _ in range(3)]
    assert list(map(len, observed)) == [8, 2, 1]
    state.wrap_batch(take, vd)
    checkpoint = tmp_path / 'model.ckpt-3.pt'
    checkpoint.write_bytes(b'checkpoint')
    state.checkpoint(checkpoint, 3)
    sidecar = json.loads((tmp_path / 'model.ckpt-3.pt.sampler.json').read_text())
    assert sidecar['batch_counts'] == {'train': 3, 'val': 1}
    expected = [state.wrap_batch(take, td) for _ in range(5)]
    expected_val = [state.wrap_batch(take, vd) for _ in range(3)]
    resumed = SamplerResume(train, val, 'batch8-identity', restart=checkpoint)
    rt, rv = Data(train, batches), Data(val, validation)
    assert [resumed.wrap_batch(take, rt) for _ in range(5)] == expected
    assert [resumed.wrap_batch(take, rv) for _ in range(3)] == expected_val
    with pytest.raises(ValueError):
        SamplerResume(train, val, 'batch1-identity', restart=checkpoint)
