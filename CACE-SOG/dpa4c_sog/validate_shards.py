"""Evaluate val checkpoint shards and merge exact full-coverage validation reports.

A shard is one checkpoint and one half-open val ordinal range. Incomplete shards
are never published as complete. Merge checks source identities and hashes,
recomputes all metrics and the existing full-val identity digest; test is never read.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from cace_sog_training.data import IndexReader
from .metrics import RunningStats
from .predict import (_check_condition, _default_loader, _evaluate, _hash_file,
                      _index_identity, _model_details, _stat, _status, _validation_frames)


def shard(database, checkpoint, start, end, output, *, loader=None):
    database = Path(database).resolve(strict=True)
    checkpoint = Path(checkpoint).resolve(strict=True)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_name(output.name + '.status.json')
    if output.exists():
        raise FileExistsError('output exists')
    with marker.open('x', encoding='utf8') as f:
        json.dump({'status': 'running'}, f)
    state = {'status': 'running', 'channel': 'sog_candidates', 'split': 'val',
             'test_split_read': False, 'condition': [0., 1.], 'checkpoint': str(checkpoint)}
    model = None
    try:
        sha, cp_stat = _hash_file(checkpoint)
        model = (loader or _default_loader)(checkpoint)
        details = _model_details(model, 'dpa')
        with IndexReader(database) as reader:
            total = reader.counts('sog_candidates', 'val')
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)) or not 0 <= start < end <= total:
                raise ValueError('invalid validation ordinal range')
            identity = _index_identity(reader.metadata)
            db_stat = _stat(reader.database)
            fd, temp = tempfile.mkstemp(prefix='.' + output.name + '.partial-', dir=output.parent)
            with os.fdopen(fd, 'w', encoding='utf8', newline='\n') as out:
                last = time.monotonic()
                for ordinal in range(start, end):
                    rec = reader.record('sog_candidates', 'val', ordinal)
                    atoms = reader.read_atoms(rec)
                    _check_condition(atoms, np.array([0., 1.]))
                    n = len(atoms)
                    force = np.asarray(atoms.arrays['REF_forces'], dtype=np.float64)
                    energy = float(atoms.info['REF_energy'])
                    if n != rec['natoms'] or n < 1 or force.shape != (n, 3) or not np.isfinite(force).all() or not math.isfinite(energy):
                        raise ValueError('invalid source labels')
                    pred_e, pred_f = _evaluate(model, details, atoms, [0., 1.], 'dpa')
                    row = {'identity': [ordinal, rec['id'], rec['sha256'], n, rec['component_id']],
                           'energy': [energy / n, pred_e / n], 'force_ref': force.tolist(), 'force_pred': pred_f}
                    out.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
                    if time.monotonic() - last >= 30:
                        print(f'validation shard {start}:{end}: {ordinal-start+1:,}/{end-start:,}', flush=True)
                        last = time.monotonic()
                out.flush(); os.fsync(out.fileno())
            reader.verify_sources()
            if _stat(reader.database) != db_stat:
                raise ValueError('index changed during validation')
        if _stat(checkpoint) != cp_stat:
            raise ValueError('checkpoint changed during validation')
        digest, _ = _hash_file(Path(temp))
        os.link(temp, output); os.unlink(temp)
        state.update(status='complete', start=start, end=end, total=total, frames=end-start,
                     index_identity=identity, checkpoint_sha256=sha, model=details, predictions_sha256=digest)
        _status(marker, state)
        return state
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:4096])
        _status(marker, state)
        raise
    finally:
        if model is not None:
            close = getattr(getattr(model, 'deep_eval', None), 'close', None)
            if callable(close): close()


def merge(database, checkpoints, shards, output):
    database = Path(database).resolve(strict=True)
    checkpoints = [Path(p).resolve(strict=True) for p in checkpoints]
    if not checkpoints or not shards:
        raise ValueError('checkpoints and shards are required')
    target = Path(output).resolve()
    if target.exists(): raise FileExistsError('output directory exists')
    target.mkdir(parents=True)
    marker = target / 'validation_status.json'
    state = {'status': 'running', 'channel': 'sog_candidates', 'split': 'val', 'test_split_read': False}
    _status(marker, state)
    try:
        cps = [_hash_file(p) for p in checkpoints]
        if len(set(x[0] for x in cps)) != len(cps): raise ValueError('duplicate checkpoints')
        with IndexReader(database) as reader:
            identity = _index_identity(reader.metadata)
            total, full_digest = _validation_frames(reader)
            db_stat = _stat(reader.database)
        grouped = {x[0]: [] for x in cps}
        for name in shards:
            path = Path(name).resolve(strict=True)
            sp = path.with_name(path.name + '.status.json')
            ss = _stat(sp); raw = sp.read_bytes()
            status = json.loads(raw)
            if _stat(sp) != ss or status.get('status') != 'complete' or status.get('channel') != 'sog_candidates' or status.get('split') != 'val' or status.get('test_split_read') is not False or status.get('condition') != [0., 1.] or status.get('total') != total or status.get('index_identity') != identity:
                raise ValueError('shard state or provenance mismatch')
            a, b = status.get('start'), status.get('end')
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (a, b)) or not 0 <= a < b <= total or status.get('frames') != b-a:
                raise ValueError('invalid shard range')
            key = status.get('checkpoint_sha256')
            if key not in grouped: raise ValueError('shard belongs to an unrequested checkpoint')
            grouped[key].append((a, b, path, status, sp, ss, hashlib.sha256(raw).hexdigest()))
        results = []
        for cp, (cp_sha, cp_stat) in zip(checkpoints, cps):
            group = sorted(grouped[cp_sha], key=lambda x: x[0])
            expected = 0
            identity_hash = hashlib.sha256()
            energy, force = RunningStats(), RunningStats()
            details = None
            for a, b, path, status, sp, ss, sh in group:
                if a != expected: raise ValueError('validation shard overlap or gap')
                if details is None: details = status['model']
                elif status['model'] != details: raise ValueError('checkpoint model metadata differs across shards')
                before = _stat(path); h = hashlib.sha256(); count = 0
                with path.open('rb') as stream:
                    for line in stream:
                        if len(line) > 64 * 1024 * 1024 or not line.endswith(b'\n'): raise ValueError('invalid shard line')
                        if count >= b-a: raise ValueError('shard has excess rows')
                        h.update(line)
                        row = json.loads(line)
                        if not isinstance(row, dict): raise ValueError('invalid shard record')
                        ident = row.get('identity')
                        if not isinstance(ident, list) or len(ident) != 5 or ident[0] != a+count or ident[0] >= b:
                            raise ValueError('shard ordinal mismatch')
                        identity_hash.update(json.dumps(ident, separators=(',', ':'), ensure_ascii=True).encode('ascii') + b'\n')
                        vals = row['energy']
                        if not isinstance(vals, list) or len(vals) != 2: raise ValueError('invalid energy')
                        energy.add([vals[0]], [vals[1]])
                        ref, pred = np.asarray(row['force_ref'], dtype=np.float64), np.asarray(row['force_pred'], dtype=np.float64)
                        if ref.shape != (ident[3], 3) or pred.shape != ref.shape: raise ValueError('invalid force shape')
                        force.add(ref, pred)
                        count += 1
                if count != b-a or h.hexdigest() != status.get('predictions_sha256') or _stat(path) != before:
                    raise ValueError('shard content or hash mismatch')
                if _stat(sp) != ss or hashlib.sha256(sp.read_bytes()).hexdigest() != sh: raise ValueError('shard status changed')
                expected = b
            if expected != total or identity_hash.hexdigest() != full_digest:
                raise ValueError('validation shards do not cover the full indexed val split')
            if _stat(cp) != cp_stat: raise ValueError('checkpoint changed')
            results.append({'checkpoint': str(cp), 'checkpoint_sha256': cp_sha, 'model': details,
                            'frames': total, 'frame_identity_sha256': full_digest,
                            'energy': energy.result('eV/atom'), 'force': force.result('eV/Å')})
        with IndexReader(database) as reader:
            reader.verify_sources()
            if _stat(reader.database) != db_stat or _index_identity(reader.metadata) != identity:
                raise ValueError('index changed during merge')
        best = min(range(len(results)), key=lambda i: results[i]['force']['mae'])
        report = {'status': 'complete', 'kind': 'dpa', 'channel': 'sog_candidates', 'split': 'val',
                  'test_split_read': False, 'index_path': str(database), 'index_identity': identity,
                  'frames_per_checkpoint': total, 'frame_identity_sha256': full_digest, 'condition': [0., 1.],
                  'condition_provenance': 'explicit model input assumption; not a measured electronic state',
                  'difference_definition': 'prediction minus DFT reference',
                  'selection': 'minimum full-validation force-component MAE; ties use input order',
                  'selected_index': best, 'selected_checkpoint': results[best]['checkpoint'],
                  'selected_checkpoint_sha256': results[best]['checkpoint_sha256'], 'candidates': results}
        _status(target / 'validation.json', report)
        state.update(status='complete', selected_checkpoint_sha256=report['selected_checkpoint_sha256'])
        _status(marker, state)
        return report
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:4096])
        _status(marker, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    s = sub.add_parser('shard')
    for name in ('database', 'checkpoint', 'output'): s.add_argument('--' + name, required=True)
    s.add_argument('--start', type=int, required=True); s.add_argument('--end', type=int, required=True)
    m = sub.add_parser('merge')
    for name in ('database', 'output'): m.add_argument('--' + name, required=True)
    m.add_argument('--checkpoint', action='append', required=True)
    m.add_argument('--shard', action='append', required=True)
    a = p.parse_args()
    result = (shard(a.database, a.checkpoint, a.start, a.end, a.output)
              if a.action == 'shard' else merge(a.database, a.checkpoint, a.shard, a.output))
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == '__main__': main()
