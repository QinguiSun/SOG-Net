import json
import fcntl
import multiprocessing
import os
import pytest
import numpy as np
from test_dpa4c_convert import make_dataset
from test_dpa4c_index_lmdb import decode
from cace_sog_training.data import IndexReader
from dpa4c_sog.index_lmdb_resume import convert


def crash_worker(db, out, stage, split):
    def fault(actual, current, seen):
        if actual == stage and current == split:
            os._exit(71)
    convert(db, out, map_size=16*1024*1024, commit_every=1, fault=fault)


@pytest.mark.parametrize('stage,split', [
    ('after_manifest_fsync', 'train'), ('after_batch_commit', 'train'),
    ('after_final_commit', 'train'), ('after_batch_commit', 'val'),
    ('after_final_commit', 'val'),
])
def test_process_death_resume_roundtrip(tmp_path, stage, split):
    lmdb = pytest.importorskip('lmdb')
    msgpack = pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path)
    out = tmp_path / 'out'
    process = multiprocessing.get_context('fork').Process(target=crash_worker, args=(db, out, stage, split))
    process.start(); process.join(10)
    assert process.exitcode == 71
    result = convert(db, out, map_size=16*1024*1024, commit_every=1)
    assert result['status'] == 'complete' and result['no_sampling'] and not result['test_split_read']
    assert result['counts'] == {'train': 3, 'val': 2}
    assert not (out / 'test.lmdb').exists()
    with IndexReader(db) as reader:
        for part in ('train', 'val'):
            env = lmdb.open(str(out / (part + '.lmdb')), readonly=True, lock=False)
            rows = [json.loads(s) for s in (out / (part + '.jsonl')).read_text().splitlines()]
            with env.begin() as tx:
                meta = msgpack.unpackb(tx.get(b'__metadata__'), raw=False)
                assert len(rows) == meta['nframes'] == result['counts'][part]
                assert tx.get(b'__resume__') is None
                for i, row in enumerate(rows):
                    frame = msgpack.unpackb(tx.get(f'{i:012d}'.encode()), raw=False)
                    source = reader.record('sog_candidates', part, row['source_ordinal'])
                    atoms = reader.read_atoms(source); order = row['canonical_to_source']
                    assert row['source_id'] == source['id'] and row['frame_sha256'] == source['sha256']
                    np.testing.assert_array_equal(decode(frame['coords']), atoms.positions[order])
                    np.testing.assert_array_equal(decode(frame['forces']), atoms.arrays['REF_forces'][order])
                    assert decode(frame['energies']) == atoms.info['REF_energy']
            env.close()
    with pytest.raises(FileExistsError):
        convert(db, out, map_size=16*1024*1024)


def test_tampered_manifest_fails_resume(tmp_path):
    pytest.importorskip('lmdb'); pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path); out = tmp_path / 'out'
    process = multiprocessing.get_context('fork').Process(target=crash_worker, args=(db, out, 'after_batch_commit', 'train'))
    process.start(); process.join(10); assert process.exitcode == 71
    manifest = out / 'train.jsonl'
    content = bytearray(manifest.read_bytes()); content[0] ^= 1; manifest.write_bytes(content)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        convert(db, out, map_size=16*1024*1024, commit_every=1)


def test_debug_limit_resume_parameters_are_immutable(tmp_path):
    pytest.importorskip('lmdb'); pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path); out = tmp_path / 'out'
    process = multiprocessing.get_context('fork').Process(target=crash_worker, args=(db, out, 'after_batch_commit', 'train'))
    process.start(); process.join(10); assert process.exitcode == 71
    with pytest.raises(ValueError, match='parameters'):
        convert(db, out, map_size=16*1024*1024, limit_per_split=2)


def test_conversion_lock_refuses_concurrent_run(tmp_path):
    pytest.importorskip('lmdb'); pytest.importorskip('msgpack')
    _, db = make_dataset(tmp_path); out = tmp_path / 'out'; out.mkdir()
    fd = os.open(out / '.converter.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='another converter'):
            convert(db, out, map_size=16*1024*1024)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)


def test_full_conversion_accepted_by_config_preparer(tmp_path):
    pytest.importorskip('lmdb'); pytest.importorskip('msgpack')
    from test_dpa4c_index_smoke import release_file
    from dpa4c_sog.prepare_lmdb import prepare
    _, db = make_dataset(tmp_path); out = tmp_path / 'out'
    release = tmp_path / 'release.json'; release_file(release)
    convert(db, out, map_size=16*1024*1024, commit_every=2)
    result = prepare(db, release, out, tmp_path / 'config', 10, 1, 42)
    assert result['status'] == 'prepared_only'
    assert result['source_counts'] == {'train': 3, 'val': 2}
