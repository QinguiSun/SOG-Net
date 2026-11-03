from types import SimpleNamespace

import pytest

from dpa4c_sog.sampler_resume_experimental import SamplerResume


class FakeIterator:
    def __init__(self):
        self.index = 0
        self._pending = None
        self._deferred_indices = None
    def _next_indices(self):
        index = self.index
        self.index += 1
        return [index]
    def take(self):
        if self._deferred_indices is None:
            value = self._next_indices()
        else:
            value = self._deferred_indices
        self._deferred_indices = self._next_indices()
        return value


class FakeData:
    def __init__(self, path):
        self._reader = SimpleNamespace(lmdb_path=str(path))
        self._sampler = SimpleNamespace(_seed=42)
        self._batch_iterator = FakeIterator()


def take(d):
    return d._batch_iterator.take()


def test_restores_train_and_validation_counts_without_prefetch_off_by_one(tmp_path):
    train, val = tmp_path / 'train.lmdb', tmp_path / 'val.lmdb'
    original = SamplerResume(train, val, 'immutable-inputs')
    train_data, val_data = FakeData(train), FakeData(val)
    assert [original.wrap_batch(take, train_data) for _ in range(3)] == [[0], [1], [2]]
    assert [original.wrap_batch(take, val_data) for _ in range(2)] == [[0], [1]]
    ckpt = tmp_path / 'model.ckpt-3.pt'; ckpt.write_bytes(b'checkpoint')
    original.checkpoint(ckpt, 3)
    want_train = [original.wrap_batch(take, train_data) for _ in range(4)]
    want_val = [original.wrap_batch(take, val_data) for _ in range(3)]
    resumed = SamplerResume(train, val, 'immutable-inputs', restart=ckpt)
    resumed_train, resumed_val = FakeData(train), FakeData(val)
    assert [resumed.wrap_batch(take, resumed_train) for _ in range(4)] == want_train
    assert [resumed.wrap_batch(take, resumed_val) for _ in range(3)] == want_val


def test_fails_closed_on_changed_checkpoint_or_fingerprint(tmp_path):
    train, val = tmp_path / 'train', tmp_path / 'val'
    ckpt = tmp_path / 'model.pt'; ckpt.write_bytes(b'a')
    state = SamplerResume(train, val, 'x'); state.checkpoint(ckpt, 3)
    with pytest.raises(ValueError): SamplerResume(train, val, 'changed', ckpt)
    ckpt.write_bytes(b'b')
    with pytest.raises(ValueError): SamplerResume(train, val, 'x', ckpt)


def test_unrecognized_data_or_existing_prefetch_is_rejected(tmp_path):
    state = SamplerResume(tmp_path / 'train', tmp_path / 'val', 'x')
    with pytest.raises(ValueError): state.wrap_batch(take, FakeData(tmp_path / 'other'))
    data = FakeData(tmp_path / 'train'); data._batch_iterator._deferred_indices = [1]
    with pytest.raises(ValueError): state.wrap_batch(take, data)
