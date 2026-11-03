"""Experimental fail-closed inspection of numbered sampler checkpoints.

This module is not a production launcher. It only inspects checkpoint pairs;
it never deletes, moves, or resumes them. Callers must lock the run, verify
the data/software fingerprint independently, and resolve newer incomplete
files before launching a writer. Network filesystem durability is not proven.
"""
import json
import os
import re
from pathlib import Path

from .sampler_resume_experimental import SamplerResume, _sha

_CHECKPOINT = re.compile(r'model\.ckpt-([1-9][0-9]*)\.pt')
_HASH = re.compile(r'[0-9a-f]{64}')


def _fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class DurableSamplerResume(SamplerResume):
    """Flush checkpoint before publishing its sidecar, then flush directory.

    The pair is not an atomic transaction; a crash can leave an orphan, which
    must be ignored and preserved until handled under the run lock.
    """
    def checkpoint(self, checkpoint, step):
        path = Path(checkpoint)
        match = _CHECKPOINT.fullmatch(path.name)
        if not match or int(match.group(1)) != step or path.is_symlink():
            raise ValueError('checkpoint name, step, or file type invalid')
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        payload = super().checkpoint(path, step)
        _fsync_dir(path.parent)
        return payload


def inspect_pairs(run, fingerprint):
    """Return candidate and all incomplete or invalid files, without mutation.

    Reject symlinks, mismatched step, fingerprint and hash, invalid sampler
    counts/seeds, and files that change while read. A selected checkpoint does
    not imply it is safe to start writing: ``conflicts`` lists *all* invalid or
    incomplete checkpoint steps at or after the selected step. A caller must
    refuse to restart while conflicts remain.
    """
    if not isinstance(fingerprint, str) or not _HASH.fullmatch(fingerprint):
        raise ValueError('expected 64-character lowercase fingerprint')
    root = Path(run).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('run must be a directory')
    steps = {}
    for p in root.iterdir():
        name = p.name[:-13] if p.name.endswith('.sampler.json') else p.name
        match = _CHECKPOINT.fullmatch(name)
        if match:
            steps.setdefault(int(match.group(1)), {})['sidecar' if p.name.endswith('.sampler.json') else 'checkpoint'] = p
    valid, invalid = [], []
    for step, pair in sorted(steps.items()):
        try:
            if set(pair) != {'checkpoint', 'sidecar'}:
                raise ValueError('incomplete checkpoint/sidecar pair')
            cp, side = pair['checkpoint'], pair['sidecar']
            if cp.is_symlink() or side.is_symlink() or not cp.is_file() or not side.is_file():
                raise ValueError('checkpoint and sidecar must be regular files')
            if side.stat().st_size > 1024 * 1024:
                raise ValueError('sidecar too large')
            before = [(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in (cp, side)]
            obj = json.loads(side.read_text(encoding='utf8'))
            if not isinstance(obj, dict) or type(obj.get('version')) is not int or obj['version'] != 1 or type(obj.get('step')) is not int or obj['step'] != step or obj.get('fingerprint') != fingerprint:
                raise ValueError('sidecar version, step or fingerprint mismatch')
            seeds, counts = obj.get('seeds'), obj.get('batch_counts')
            if not isinstance(counts, dict) or set(counts) != {'train', 'val'} or any(type(n) is not int or n < 0 for n in counts.values()):
                raise ValueError('invalid sampler counts')
            if not isinstance(seeds, dict) or set(seeds) != {'train', 'val'} or any(v is not None and (type(v) is not int or v < 0) for v in seeds.values()):
                raise ValueError('invalid sampler seeds')
            if any(counts[k] and seeds[k] is None for k in counts):
                raise ValueError('missing sampler seed')
            digest = _sha(cp)
            if digest != obj.get('checkpoint_sha256'):
                raise ValueError('checkpoint SHA mismatch')
            after = [(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in (cp, side)]
            if before != after:
                raise ValueError('pair changed during inspection')
            valid.append({'step': step, 'checkpoint': str(cp), 'checkpoint_sha256': digest})
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            invalid.append({'step': step, 'error': str(exc), 'files': sorted(p.name for p in pair.values())})
    selected = valid[-1] if valid else None
    conflicts = [x for x in invalid if selected is None or x['step'] >= selected['step']]
    return {'selected': selected, 'valid': valid, 'invalid': invalid, 'conflicts': conflicts,
            'safe_to_launch': bool(selected) and not conflicts}
