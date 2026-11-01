"""Merge completed ordinal shards only after validating full paired test coverage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .predict import _hash_file, _stat, _status


def merge(shards, output):
    output = Path(output).expanduser().resolve()
    if not shards:
        raise ValueError('no shards provided')
    output.parent.mkdir(parents=True, exist_ok=True)
    marker = output.with_name(output.name + '.status.json')
    if output.exists():
        raise ValueError('output already exists')
    with marker.open('x', encoding='utf8') as f:
        json.dump({'status': 'running'}, f)
    state = {'status': 'running', 'channel': 'sog_candidates', 'split': 'test'}
    try:
        entries = []
        for path in shards:
            path = Path(path).expanduser().resolve(strict=True)
            sp = path.with_name(path.name + '.status.json')
            raw = sp.read_bytes()
            status = json.loads(raw)
            if status.get('status') != 'complete' or status.get('split') != 'test' or status.get('channel') != 'sog_candidates':
                raise ValueError(f'incomplete or invalid shard: {path}')
            start, end, total = (status.get(k) for k in ('ordinal_start', 'ordinal_end', 'test_total_frames'))
            if any(isinstance(x, bool) or not isinstance(x, int) for x in (start, end, total)) or not 0 <= start < end <= total:
                raise ValueError('invalid shard interval')
            if status.get('expected_frames') != end-start or status.get('frames_written') != end-start:
                raise ValueError('incomplete shard frame count')
            entries.append((start, end, path, status, hashlib.sha256(raw).hexdigest(), _stat(sp)))
        entries.sort(key=lambda e: e[0])
        first = entries[0][3]
        fingerprint = lambda s: (s['index_identity'], s['test_total_frames'], s['charge_spin'],
            s['checkpoints']['pretrained']['sha256'], s['checkpoints']['finetuned']['sha256'],
            s['validation_provenance']['finetuned']['sha256'])
        reference = fingerprint(first)
        expected = 0
        for start, end, path, status, _, _ in entries:
            if start != expected or fingerprint(status) != reference:
                raise ValueError('shards overlap, have gaps, or have inconsistent provenance')
            expected = end
        if expected != first['test_total_frames']:
            raise ValueError('shards do not cover the entire test split')
        fd, temp = tempfile.mkstemp(prefix='.' + output.name + '.merge-', dir=output.parent)
        temp = Path(temp)
        digest = hashlib.sha256()
        with os.fdopen(fd, 'wb') as out:
            for start, end, path, status, status_sha, status_stat in entries:
                before = _stat(path)
                shard_digest = hashlib.sha256()
                count = 0
                with path.open('rb') as source:
                    for raw in source:
                        if len(raw) > 64*1024*1024 or not raw.endswith(b'\n'):
                            raise ValueError('invalid or oversized shard row')
                        row = json.loads(raw)
                        ordinal = start + count
                        if ordinal >= end or row.get('frame_key') != f'sog_candidates/test/{ordinal}' or row.get('split') != 'test':
                            raise ValueError('shard row ordering or split differs')
                        shard_digest.update(raw); digest.update(raw); out.write(raw)
                        count += 1
                if count != end-start or shard_digest.hexdigest() != status['predictions_sha256'] or _stat(path) != before:
                    raise ValueError(f'shard contents changed or hash differs: {path}')
                sp = path.with_name(path.name + '.status.json')
                if _stat(sp) != status_stat or hashlib.sha256(sp.read_bytes()).hexdigest() != status_sha:
                    raise ValueError('shard status changed during merge')
            out.flush(); os.fsync(out.fileno())
        os.link(temp, output)
        temp.unlink()
        state.update(status='complete', is_shard=False, ordinal_start=0, ordinal_end=expected,
            test_total_frames=expected, expected_frames=expected, frames_written=expected,
            index_identity=first['index_identity'], charge_spin=first['charge_spin'],
            condition_provenance=first['condition_provenance'], checkpoints=first['checkpoints'],
            validation_provenance=first['validation_provenance'], predictions_sha256=digest.hexdigest(),
            shards=[{'path': str(e[2]), 'start': e[0], 'end': e[1], 'status_sha256': e[4]} for e in entries])
        _status(marker, state)
        return state
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}'[:4096])
        _status(marker, state)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards', nargs='+', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(merge(args.shards, args.output), indent=2))


if __name__ == '__main__':
    main()
