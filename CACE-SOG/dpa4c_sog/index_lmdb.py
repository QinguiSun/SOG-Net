"""Research converter: index to train/val LMDB; no test frames are read.

Format follows the installed SCNet dpdata LMDB writer. This is NOT evidence
that the SCNet trainer can use the output until an on-device round-trip test.
Metadata and DeepMD sampler indexes grow O(number of frames), despite framewise
I/O. A per-split frame limit bounds metadata size; map-size bounds LMDB growth.
Every frame is assigned to one logical sampling system per split, so sampling
does not preserve the original material/trajectory weighting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cace_sog_training.data import IndexReader
from .convert import TYPE_MAP, _check_condition, _write_json
from .index_smoke import _identity


def encoded(value, dtype):
    arr = np.asarray(value, dtype=dtype)
    return {'type': str(arr.dtype), 'shape': list(arr.shape), 'data': arr.tobytes(order='C')}


def frame_payload(atoms):
    n = len(atoms)
    order = np.argsort(atoms.numbers, kind='stable')
    types = np.asarray(atoms.numbers[order] - 1, dtype=np.int32)
    if n < 1 or np.any(types < 0) or np.any(types >= len(TYPE_MAP)):
        raise ValueError('invalid atom types')
    _check_condition(atoms.info)
    if not np.asarray(atoms.pbc).all():
        raise ValueError('only fully periodic frames are supported')
    coords = np.asarray(atoms.positions[order], dtype=np.float64)
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    force = np.asarray(atoms.arrays['REF_forces'][order], dtype=np.float64)
    energy = np.asarray(atoms.info['REF_energy'], dtype=np.float64)
    if coords.shape != (n, 3) or force.shape != (n, 3) or cell.shape != (3, 3) or energy.shape != ():
        raise ValueError('invalid frame array shape')
    if not all(np.isfinite(x).all() for x in (coords, cell, force, energy)):
        raise ValueError('non-finite frame value')
    payload = {'atom_types': encoded(types, np.int32),
               'atom_numbs': np.bincount(types, minlength=len(TYPE_MAP)).tolist(),
               'coords': encoded(coords, np.float64), 'cells': encoded(cell, np.float64),
               'forces': encoded(force, np.float64), 'energies': encoded(energy, np.float64)}
    return payload, order.tolist()


def convert(database, output, max_frames_per_split=4000000, map_size=32 * 1024**3,
            commit_every=1000, limit_per_split=None, *, lmdb_module=None, msgpack_module=None):
    if max_frames_per_split < 1 or map_size < 1 or commit_every < 1:
        raise ValueError('limits must be positive')
    if limit_per_split is not None and limit_per_split < 1:
        raise ValueError('limit_per_split must be positive')
    if lmdb_module is None:
        import lmdb as lmdb_module
    if msgpack_module is None:
        import msgpack as msgpack_module
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    with IndexReader(database) as reader:
        source_counts = {s: reader.counts('sog_candidates', s) for s in ('train', 'val')}
        counts = {s: min(n, limit_per_split) if limit_per_split else n for s, n in source_counts.items()}
        if any(n < 1 or n > max_frames_per_split for n in counts.values()):
            raise ValueError('split empty or exceeds --max-frames-per-split')
        source_root = Path(reader.metadata['config']['export_dir']).resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError('output must not be inside source export')
        output.mkdir(parents=True)
        status = {'status': 'partial', 'channel': 'sog_candidates', 'counts': counts,
                  'source_counts': source_counts, 'debug_only': limit_per_split is not None,
                  'limit_per_split': limit_per_split,
                  'test_split_read': False, 'no_sampling': limit_per_split is None, 'condition': [0, 1],
                  'condition_is_not_measured_dft_state': True,
                  'sampling_system': 'one logical system per split; frame-level sampling',
                  'source_index_metadata_sha256': _identity(reader.metadata),
                  'max_frames_per_split': max_frames_per_split, 'lmdb_map_size': map_size,
                  'compatibility': 'candidate format; real SCNet reader and training roundtrip required'}
        _write_json(output / 'status.json', status)
        try:
            for split in ('train', 'val'):
                env = lmdb_module.open(str(output / (split + '.lmdb')), map_size=map_size,
                                       subdir=True, create=True, max_dbs=1)
                nlocs, seen = [], 0
                start = time.monotonic()
                try:
                    with (output / (split + '.jsonl')).open('x', encoding='utf8') as manifest:
                        tx = env.begin(write=True)
                        try:
                            for row in reader.iter_records(('sog_candidates',), split):
                                if seen == counts[split]:
                                    break
                                if seen >= max_frames_per_split:
                                    raise ValueError('frame limit exceeded')
                                atoms = reader.read_atoms(row)
                                payload, permutation = frame_payload(atoms)
                                key = format(seen, '012d').encode('ascii')
                                if not tx.put(key, msgpack_module.packb(payload, use_bin_type=True), overwrite=False):
                                    raise ValueError('duplicate LMDB key')
                                manifest.write(json.dumps({'key': key.decode(), 'source_id': int(row['id']),
                                    'source_ordinal': int(row['ordinal']), 'frame_sha256': row['sha256'],
                                    'component_id': row['component_id'], 'split': split,
                                    'canonical_to_source': permutation}, separators=(',', ':')) + '\n')
                                nlocs.append(len(atoms)); seen += 1
                                if seen % commit_every == 0:
                                    tx.commit(); tx = env.begin(write=True)
                                if time.monotonic() - start >= 30:
                                    print(f'{split}: {seen:,}/{counts[split]:,}', flush=True); start = time.monotonic()
                            if seen != counts[split]:
                                raise ValueError('index count mismatch')
                            meta = {'nframes': seen, 'frame_idx_fmt': '012d', 'type_map': list(TYPE_MAP),
                                    'frame_nlocs': nlocs, 'frame_system_ids': [0] * seen}
                            tx.put(b'__metadata__', msgpack_module.packb(meta, use_bin_type=True))
                            tx.commit(); tx = None
                        finally:
                            if tx is not None:
                                tx.abort()
                    env.sync()
                finally:
                    env.close()
                with (output / (split + '.jsonl')).open('rb') as f:
                    h = hashlib.sha256()
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        h.update(block)
                status[split + '_manifest_sha256'] = h.hexdigest()
                with (output / (split + '.lmdb') / 'data.mdb').open('rb') as f:
                    h = hashlib.sha256()
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                        h.update(block)
                status[split + '_lmdb_sha256'] = h.hexdigest()
            reader.verify_sources()
            status['status'] = 'complete'
            _write_json(output / 'status.json', status)
            return status
        except BaseException as exc:
            status['error'] = str(exc)[:2048]
            _write_json(output / 'status.json', status)
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-frames-per-split', type=int, default=4000000)
    p.add_argument('--map-size', type=int, default=32 * 1024**3)
    p.add_argument('--commit-every', type=int, default=1000)
    p.add_argument('--limit-per-split', type=int, help='first N records per split for debugging only')
    args = p.parse_args()
    print(json.dumps(convert(args.database, args.output, args.max_frames_per_split,
                             args.map_size, args.commit_every, args.limit_per_split), indent=2))


if __name__ == '__main__':
    main()
