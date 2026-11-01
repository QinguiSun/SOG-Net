"""Resumable research conversion of an ELEMENTA index to train/val LMDB.

Each committed LMDB batch includes a marker pointing to an fsynced manifest
prefix. Uncommitted manifest tails are truncated on recovery. Frames are not
sampled by default. One logical system per split changes weighting relative to
material-level sampling. LMDB metadata and training sampler use O(N) memory;
neither full-scale memory safety nor installed DeePMD training is certified.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from cace_sog_training.data import IndexReader
from .convert import TYPE_MAP, _write_json
from .index_lmdb import frame_payload
from .index_smoke import _identity
from .prepare import sha256

MARKER = b'__resume__'
META = b'__metadata__'


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _pack(msgpack, value):
    return msgpack.packb(value, use_bin_type=True)


def _recover_prefix(path, offset, expected_hash, count):
    h = hashlib.sha256()
    nlocs = []
    with path.open('r+b') as f:
        if os.fstat(f.fileno()).st_size < offset:
            raise ValueError('manifest is shorter than committed marker')
        for index in range(count):
            line = f.readline()
            if not line.endswith(b'\n') or f.tell() > offset:
                raise ValueError('manifest committed prefix is malformed')
            h.update(line)
            row = json.loads(line)
            if row['key'] != format(index, '012d') or int(row['natoms']) < 1:
                raise ValueError('manifest frame sequence invalid')
            nlocs.append(int(row['natoms']))
        if f.tell() != offset or h.hexdigest() != expected_hash:
            raise ValueError('committed manifest prefix hash or length mismatch')
        f.truncate(offset)
        f.flush(); os.fsync(f.fileno())
    return h, nlocs


def _done_meta(env, expected, msgpack):
    with env.begin() as txn:
        raw = txn.get(META)
        if raw is None:
            return None
        meta = msgpack.unpackb(raw, raw=False)
        if (meta.get('nframes') != expected or meta.get('type_map') != list(TYPE_MAP)
                or len(meta.get('frame_nlocs', [])) != expected
                or meta.get('frame_system_ids') != [0] * expected
                or txn.get(MARKER) is not None
                or txn.stat()['entries'] != expected + 1):
            raise ValueError('completed LMDB metadata/count mismatch')
        for i in range(expected):
            if txn.get(format(i, '012d').encode()) is None:
                raise ValueError('completed LMDB has missing frame key')
        return meta


def convert(database, output, max_frames_per_split=4000000, map_size=32 * 1024**3,
            commit_every=1000, limit_per_split=None, *, fault=None):
    import lmdb
    import msgpack
    if min(max_frames_per_split, map_size, commit_every) < 1 or (limit_per_split is not None and limit_per_split < 1):
        raise ValueError('limits must be positive')
    fault = fault or (lambda stage, split, count: None)
    out = Path(output).expanduser().resolve()
    with IndexReader(database) as reader:
        source_root = Path(reader.metadata['config']['export_dir']).resolve()
        if out == source_root or source_root in out.parents:
            raise ValueError('output must not be inside the source export directory')
        source_counts = {s: reader.counts('sog_candidates', s) for s in ('train', 'val')}
        counts = {s: min(n, limit_per_split) if limit_per_split else n for s, n in source_counts.items()}
        if any(n < 1 or n > max_frames_per_split for n in counts.values()):
            raise ValueError('split empty or exceeds frame limit')
        reader.verify_sources()
        identity = _identity(reader.metadata)
        params = {'version': 1, 'source_index_metadata_sha256': identity, 'source_counts': source_counts,
                  'counts': counts, 'limit_per_split': limit_per_split, 'max_frames_per_split': max_frames_per_split}
        out.mkdir(parents=True, exist_ok=True)
        lockfd = os.open(out / '.converter.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(lockfd)
            raise RuntimeError('another converter holds the output lock')
        try:
            param_path = out / 'resume_parameters.json'
            if param_path.exists():
                if json.loads(param_path.read_text()) != params:
                    raise ValueError('resume parameters or source index identity differ')
            else:
                if any(p.name != '.converter.lock' for p in out.iterdir()):
                    raise ValueError('nonempty directory is not a resumable conversion')
                _write_json(param_path, params); _sync_directory(out)
            status_path = out / 'status.json'
            if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'complete':
                raise FileExistsError('conversion already complete; no overwrite')
            status = {'status': 'partial', 'channel': 'sog_candidates', 'counts': counts,
                      'source_counts': source_counts, 'debug_only': limit_per_split is not None,
                      'limit_per_split': limit_per_split, 'no_sampling': limit_per_split is None,
                      'test_split_read': False, 'condition': [0, 1],
                      'condition_is_not_measured_dft_state': True, 'source_index_metadata_sha256': identity,
                      'sampling_system': 'one logical system per split; frame-level sampling',
                      'max_frames_per_split': max_frames_per_split, 'lmdb_map_size': map_size,
                      'compatibility': 'research format; real SCNet reader and training roundtrip required'}
            _write_json(status_path, status)
            for split in ('train', 'val'):
                path = out / (split + '.lmdb')
                manifest = out / (split + '.jsonl')
                env = lmdb.open(str(path), map_size=map_size, subdir=True, create=True, max_dbs=1)
                try:
                    complete = _done_meta(env, counts[split], msgpack)
                    if complete is None:
                        if not manifest.exists():
                            with manifest.open('xb') as f:
                                f.flush(); os.fsync(f.fileno())
                            _sync_directory(out)
                        with env.begin(write=True) as tx:
                            raw = tx.get(MARKER)
                            if raw is None:
                                if tx.stat()['entries']:
                                    raise ValueError('partial database has no transactional marker')
                                mark = {'count': 0, 'offset': 0, 'sha256': hashlib.sha256().hexdigest(),
                                        'identity': identity, 'split': split}
                                tx.put(MARKER, _pack(msgpack, mark))
                            else:
                                mark = msgpack.unpackb(raw, raw=False)
                        if mark['identity'] != identity or mark['split'] != split or not 0 <= mark['count'] <= counts[split]:
                            raise ValueError('invalid committed progress marker')
                        seen = int(mark['count'])
                        with env.begin() as tx:
                            if tx.stat()['entries'] != seen + 1:
                                raise ValueError('partial LMDB record count mismatch')
                        digest, nlocs = _recover_prefix(manifest, mark['offset'], mark['sha256'], seen)
                        print(f'{split}: resume {seen:,}/{counts[split]:,}', flush=True)
                        with manifest.open('ab') as mf:
                            tx = env.begin(write=True)
                            try:
                                batch = 0
                                for row in reader.iter_records(('sog_candidates',), split):
                                    ordinal = int(row['ordinal'])
                                    if ordinal < seen:
                                        continue
                                    if seen >= counts[split]:
                                        break
                                    if ordinal != seen:
                                        raise ValueError('source index ordinal discontinuity')
                                    atoms = reader.read_atoms(row)
                                    payload, order = frame_payload(atoms)
                                    key = format(seen, '012d')
                                    if not tx.put(key.encode(), _pack(msgpack, payload), overwrite=False):
                                        raise ValueError('duplicate LMDB frame key')
                                    record = {'key': key, 'source_id': int(row['id']), 'source_ordinal': ordinal,
                                              'frame_sha256': row['sha256'], 'component_id': row['component_id'],
                                              'split': split, 'natoms': len(atoms), 'canonical_to_source': order}
                                    line = (json.dumps(record, separators=(',', ':')) + '\n').encode()
                                    mf.write(line); digest.update(line); nlocs.append(len(atoms)); seen += 1; batch += 1
                                    if batch == commit_every or seen == counts[split]:
                                        mf.flush(); os.fsync(mf.fileno())
                                        fault('after_manifest_fsync', split, seen)
                                        mark = {'count': seen, 'offset': mf.tell(), 'sha256': digest.hexdigest(),
                                                'identity': identity, 'split': split}
                                        tx.put(MARKER, _pack(msgpack, mark)); tx.commit(); tx = None
                                        fault('after_batch_commit', split, seen)
                                        print(f'{split}: {seen:,}/{counts[split]:,}', flush=True)
                                        tx = env.begin(write=True); batch = 0
                                if seen != counts[split] or batch:
                                    raise ValueError('source count mismatch or uncommitted batch')
                                meta = {'nframes': seen, 'frame_idx_fmt': '012d', 'type_map': list(TYPE_MAP),
                                        'frame_nlocs': nlocs, 'frame_system_ids': [0] * seen,
                                        'manifest_sha256': digest.hexdigest()}
                                tx.put(META, _pack(msgpack, meta)); tx.delete(MARKER); tx.commit(); tx = None
                                fault('after_final_commit', split, seen)
                            finally:
                                if tx is not None:
                                    tx.abort()
                        env.sync()
                        complete = _done_meta(env, counts[split], msgpack)
                    if complete is None or not manifest.is_file():
                        raise ValueError('completed split missing metadata or manifest')
                    # Verify manifest still maps exactly onto the finished LMDB.
                    with manifest.open('rb') as f:
                        h = hashlib.sha256(); n = 0
                        for line in f:
                            row = json.loads(line); h.update(line)
                            if row['key'] != format(n, '012d') or row['split'] != split or row['natoms'] != complete['frame_nlocs'][n]:
                                raise ValueError('completed manifest sequence mismatch')
                            n += 1
                        if n != counts[split]:
                            raise ValueError('completed manifest frame count mismatch')
                        if h.hexdigest() != complete.get('manifest_sha256'):
                            raise ValueError('completed manifest hash mismatch')
                    status[split + '_manifest_sha256'] = h.hexdigest()
                finally:
                    env.close()
                status[split + '_lmdb_sha256'] = sha256(path / 'data.mdb')
                _write_json(status_path, status)
            reader.verify_sources()
            status['status'] = 'complete'
            _write_json(status_path, status); _sync_directory(out)
            return status
        finally:
            fcntl.flock(lockfd, fcntl.LOCK_UN)
            os.close(lockfd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-frames-per-split', type=int, default=4000000)
    p.add_argument('--map-size', type=int, default=32 * 1024**3)
    p.add_argument('--commit-every', type=int, default=1000)
    p.add_argument('--limit-per-split', type=int)
    a = p.parse_args()
    print(json.dumps(convert(a.database, a.output, a.max_frames_per_split, a.map_size,
                             a.commit_every, a.limit_per_split), indent=2))


if __name__ == '__main__':
    main()
