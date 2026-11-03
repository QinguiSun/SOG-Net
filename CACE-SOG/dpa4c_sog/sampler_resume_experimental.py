"""Experimental single-rank LMDB sampler checkpoint wrapper.

Not a production launcher: the SCNet DeepMD build and training path must be
validated against uninterrupted runs before using it. The caller supplies a
fingerprint identifying immutable data, normalized configuration, seed,
software build and worker topology; mismatches fail closed.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class SamplerResume:
    """Track completed get_batch calls; restore before the next call.

    Internal _next_indices access is tied to the inspected DeepMD build. It
    replays sampler indices, not actual training, and can take O(saved steps).
    """
    def __init__(self, train_path, val_path, fingerprint, restart=None):
        self.paths = {str(Path(p).resolve()): k for p, k in ((train_path, 'train'), (val_path, 'val'))}
        if len(self.paths) != 2 or not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError('distinct dataset paths and a fingerprint are required')
        self.fingerprint = fingerprint
        self.seeds = {'train': None, 'val': None}
        self.counts = {'train': 0, 'val': 0}
        self.restore = {'train': 0, 'val': 0}
        self.initialized = {}
        self.restart_step = None
        if restart is not None:
            checkpoint = Path(restart).resolve(strict=True)
            sidecar = Path(str(checkpoint) + '.sampler.json')
            payload = json.loads(sidecar.read_text(encoding='utf8'))
            if payload.get('version') != 1 or payload.get('fingerprint') != fingerprint or payload.get('checkpoint_sha256') != _sha(checkpoint):
                raise ValueError('sampler checkpoint fingerprint/hash mismatch')
            if type(payload.get('step')) is not int or payload['step'] < 1:
                raise ValueError('invalid checkpoint step in sidecar')
            self.restart_step = payload['step']
            counts = payload.get('batch_counts')
            if not isinstance(counts, dict) or set(counts) != {'train', 'val'} or any(type(v) is not int or v < 0 for v in counts.values()):
                raise ValueError('invalid saved sampler counts')
            self.counts = dict(counts)
            self.restore = dict(counts)
            seeds = payload.get('seeds')
            if not isinstance(seeds, dict) or set(seeds) != {'train', 'val'} or any(v is not None and (type(v) is not int or v < 0) for v in seeds.values()):
                raise ValueError('invalid saved sampler seeds')
            if any(counts[k] and seeds[k] is None for k in counts):
                raise ValueError('consumed split has no sampler seed')
            self.seeds = dict(seeds)

    def wrap_batch(self, original, data_system, *args, **kwargs):
        name = self.paths.get(str(Path(data_system._reader.lmdb_path).resolve()))
        if name is None:
            raise ValueError('unexpected LMDB source')
        seed = getattr(data_system._sampler, '_seed', None)
        if type(seed) is not int or seed < 0:
            raise ValueError('deterministic sampler seed is required')
        if self.seeds[name] is not None and self.seeds[name] != seed:
            raise ValueError('sampler seed differs')
        self.seeds[name] = seed
        iterator = data_system._batch_iterator
        if name in self.initialized and self.initialized[name] is not data_system:
            raise ValueError('multiple data systems for the same split are unsupported')
        if name not in self.initialized:
            if getattr(iterator, '_pending', None) is not None or getattr(iterator, '_deferred_indices', None) is not None:
                raise ValueError('sampler was already prefetched before restore')
            for _ in range(self.restore[name]):
                iterator._next_indices()
            self.initialized[name] = data_system
        batch = original(data_system, *args, **kwargs)
        self.counts[name] += 1
        return batch

    def checkpoint(self, checkpoint, step):
        checkpoint = Path(checkpoint).resolve(strict=True)
        payload = {'version': 1, 'fingerprint': self.fingerprint, 'seeds': self.seeds.copy(), 'step': int(step),
                   'checkpoint_sha256': _sha(checkpoint), 'batch_counts': self.counts.copy()}
        path = Path(str(checkpoint) + '.sampler.json')
        if path.exists(): raise FileExistsError('sampler sidecar already exists')
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
        try:
            with os.fdopen(fd, 'w', encoding='utf8') as f:
                json.dump(payload, f, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
            os.link(tmp, path)
            os.unlink(tmp)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return payload


def install(data_system_class, trainer_class, state):
    """Patch only this Python process; single task and rank required."""
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('only single rank is supported')
    batch = data_system_class.get_batch
    write = trainer_class._write_checkpoint
    run = trainer_class.run
    def get_batch(instance, *args, **kwargs):
        return state.wrap_batch(batch, instance, *args, **kwargs)
    def write_checkpoint(instance, checkpoint, *, step, **kwargs):
        if getattr(instance, 'rank', 0) != 0 or getattr(instance, 'multi_task', False):
            raise ValueError('only single task and rank are supported')
        checkpoint = Path(checkpoint)
        if checkpoint.name != f'model.ckpt-{step}.pt' or kwargs.get('include_optimizer', True) is not True:
            raise ValueError('only numbered optimizer-bearing checkpoints are supported')
        if checkpoint.exists() or Path(str(checkpoint) + '.sampler.json').exists():
            raise FileExistsError('refusing to overwrite a checkpoint or sampler sidecar')
        result = write(instance, checkpoint, step=step, **kwargs)
        state.checkpoint(checkpoint, step)
        return result
    def checked_run(instance, *args, **kwargs):
        if state.restart_step is not None and getattr(instance, 'start_step', None) != state.restart_step:
            raise ValueError('loaded trainer step disagrees with sampler sidecar')
        return run(instance, *args, **kwargs)
    data_system_class.get_batch = get_batch
    trainer_class._write_checkpoint = write_checkpoint
    trainer_class.run = checked_run
