import json
import pickle
import numpy as np
import pytest
from pathlib import Path
from cace_sog_training.data import IndexReader
from test_dpa4c_convert import make_dataset
from dpa4c_sog.index_lmdb import convert


class FakeMsgpack:
    @staticmethod
    def packb(value, use_bin_type=True):
        return pickle.dumps(value)


class FakeTxn:
    def __init__(self, database): self.database = database
    def put(self, key, value, overwrite=True):
        if not overwrite and key in self.database: return False
        self.database[key] = value
        return True
    def commit(self): pass
    def abort(self): pass


class FakeEnv:
    def __init__(self): self.database = {}
    def begin(self, write=False): return FakeTxn(self.database)
    def sync(self): pass
    def close(self): pass


class FakeLMDB:
    def __init__(self): self.environments = {}
    def open(self, path, **kwargs):
        Path(path).mkdir()
        (Path(path) / 'data.mdb').write_bytes(b'fake LMDB')
        self.environments[path] = FakeEnv()
        return self.environments[path]


def decode(x):
    return np.frombuffer(x['data'], dtype=x['type']).reshape(x['shape'])


def test_train_val_only_and_source_mapping(tmp_path):
    _, db = make_dataset(tmp_path)
    fake = FakeLMDB()
    out = tmp_path / 'lmdb'
    result = convert(db, out, commit_every=2, lmdb_module=fake, msgpack_module=FakeMsgpack)
    assert result['counts'] == {'train': 3, 'val': 2}
    assert result['test_split_read'] is False
    assert result['status'] == 'complete'
    assert not (out / 'test.jsonl').exists()
    for split, n in result['counts'].items():
        data = fake.environments[str(out / (split + '.lmdb'))].database
        meta = pickle.loads(data[b'__metadata__'])
        assert meta['nframes'] == n
        assert meta['frame_system_ids'] == [0] * n
        rows = [json.loads(s) for s in (out / (split + '.jsonl')).read_text().splitlines()]
        assert len(rows) == n
        with IndexReader(db) as reader:
            for i, row in enumerate(rows):
                frame = pickle.loads(data[f'{i:012d}'.encode()])
                source = reader.record('sog_candidates', split, row['source_ordinal'])
                atoms = reader.read_atoms(source)
                order = row['canonical_to_source']
                assert row['source_id'] == source['id'] and row['frame_sha256'] == source['sha256']
                np.testing.assert_array_equal(decode(frame['atom_types']), atoms.numbers[order] - 1)
                np.testing.assert_array_equal(decode(frame['coords']), atoms.positions[order])
                np.testing.assert_array_equal(decode(frame['forces']), atoms.arrays['REF_forces'][order])
                np.testing.assert_array_equal(decode(frame['cells']), atoms.cell.array)
                assert decode(frame['energies']) == atoms.info['REF_energy']
                assert row['split'] == split


def test_frame_limit_and_existing_output(tmp_path):
    _, db = make_dataset(tmp_path)
    fake = FakeLMDB()
    with pytest.raises(ValueError, match='exceeds'):
        convert(db, tmp_path / 'too-big', max_frames_per_split=2, lmdb_module=fake, msgpack_module=FakeMsgpack)
    out = tmp_path / 'existing'
    out.mkdir()
    with pytest.raises(FileExistsError):
        convert(db, out, lmdb_module=fake, msgpack_module=FakeMsgpack)


def test_real_lmdb_msgpack_round_trip_and_debug_limit(tmp_path):
    lmdb = pytest.importorskip('lmdb')
    msgpack = pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path)
    out = tmp_path / 'real'
    result = convert(db, out, map_size=16 * 1024 * 1024,
                     commit_every=1, limit_per_split=2)
    assert result['debug_only'] and not result['no_sampling']
    assert result['counts'] == {'train': 2, 'val': 2}
    assert result['source_counts'] == {'train': 3, 'val': 2}
    assert not (out / 'test.lmdb').exists()
    with IndexReader(db) as reader:
        for split in ('train', 'val'):
            env = lmdb.open(str(out / (split + '.lmdb')), readonly=True, lock=False)
            with env.begin() as tx:
                meta = msgpack.unpackb(tx.get(b'__metadata__'), raw=False)
                assert meta['nframes'] == 2 and meta['frame_system_ids'] == [0, 0]
                rows = [json.loads(s) for s in (out / (split + '.jsonl')).read_text().splitlines()]
                for i, row in enumerate(rows):
                    record = reader.record('sog_candidates', split, row['source_ordinal'])
                    atoms = reader.read_atoms(record)
                    frame = msgpack.unpackb(tx.get(f'{i:012d}'.encode()), raw=False)
                    order = row['canonical_to_source']
                    assert np.all(atoms.pbc)
                    assert decode(frame['atom_types']).dtype == np.dtype('int32')
                    assert decode(frame['coords']).dtype == np.dtype('float64')
                    np.testing.assert_array_equal(decode(frame['atom_types']), atoms.numbers[order] - 1)
                    np.testing.assert_array_equal(decode(frame['coords']), atoms.positions[order])
                    np.testing.assert_array_equal(decode(frame['forces']), atoms.arrays['REF_forces'][order])
                    np.testing.assert_array_equal(decode(frame['cells']), atoms.cell.array)
                    assert decode(frame['energies']) == atoms.info['REF_energy']
            env.close()
