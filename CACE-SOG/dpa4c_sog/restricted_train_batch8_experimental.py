"""Experimental single-rank sampler-aware DeepMD batch-8 entry point.

Not the production launcher. Validate against SCNet before use. This protocol
restores optimizer and sampler progress but does not save RNG state and makes
no bitwise-reproducibility claim. Checkpoint and sidecar are not one atomic
filesystem transaction; incomplete newer files stop recovery.
If a final pair exists without a completion marker, refuse restart for manual
review. The checked source files do not bind every transitive dependency.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import shutil
import stat
import sys

from .sampler_recovery import DurableSamplerResume, inspect_pairs, _fsync_dir
from .sampler_resume_experimental import _sha, install

_HASH = re.compile(r'[a-f0-9]{64}')


def validate_run_symlinks(run):
    """Allow only DeepMD's direct, local checkpoint publication links.

    A crash after saving a new pair but before publication may leave both
    links pointing at the preceding pair. Selection uses numbered pair hashes,
    not these links; the pointers need not identify the latest checkpoint.
    """
    patterns = {
        'model.ckpt.pt': re.compile(r'model\.ckpt-([1-9][0-9]*)\.pt'),
        'model.ckpt.pt.sampler.json': re.compile(r'model\.ckpt-([1-9][0-9]*)\.pt\.sampler\.json'),
    }
    steps = {}
    for item in Path(run).iterdir():
        if not item.is_symlink():
            continue
        pattern = patterns.get(item.name)
        target = os.readlink(item)
        match = pattern.fullmatch(target) if pattern is not None else None
        if match is None:
            raise ValueError('unexpected or unsafe symlink in run directory: ' + item.name)
        target_path = item.parent / target
        try:
            target_mode = os.lstat(target_path).st_mode
        except OSError as exc:
            raise ValueError('missing symlink target: ' + item.name) from exc
        if not stat.S_ISREG(target_mode):
            raise ValueError('symlink target must be a direct regular file: ' + item.name)
        steps[item.name] = int(match.group(1))
    if len(steps) == 2 and len(set(steps.values())) != 1:
        raise ValueError('checkpoint publication symlink steps disagree')


def fingerprint(info):
    return hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_config(cfg, train, val):
    """Limit the experiment to batch-8 training and batch-1 validation."""
    t = cfg['training']
    for name, path in (('training_data', train), ('validation_data', val)):
        x = t[name]
        expected_batch = 8 if name == 'training_data' else 1
        if Path(x['systems']).resolve() != path.resolve() or type(x['batch_size']) is not int or x['batch_size'] != expected_batch:
            raise ValueError('configuration data path or batch size mismatch')
    if (t.get('enable_compile') is not False or t.get('enable_ema', False) is not False
            or t.get('zero_stage', 0) != 0 or t.get('change_bias_after_training', False) is not False
            or t.get('ckpt_keep_ratio') is not None or cfg['model'].get('model_dict') is not None
            or type(t.get('num_steps')) is not int or t['num_steps'] < 1
            or t.get('seed') != 42 or type(t.get('save_freq')) is not int or t['save_freq'] < 1
            or t['training_data'].get('auto_prob_style') != 'prob_sys_size'
            or t['validation_data'].get('numb_batch') != 1):
        raise ValueError('unsupported training settings')
    if cfg.get('model', {}).get('descriptor', {}).get('type') != 'dpa4c':
        raise ValueError('only single-task DPA4C is supported')
    if t.get('max_ckpt_keep', 3) < 2:
        raise ValueError('retain at least two checkpoints')
    if (t.get('save_dir') not in (None, '', '.') or t.get('save_ckpt', 'model.ckpt') != 'model.ckpt'
            or t.get('disp_file', 'lcurve.out') != 'lcurve.out'
            or t.get('profiling', False) is not False or t.get('tensorboard', False) is not False):
        raise ValueError('unsupported output path or profiling option')
    validating = cfg.get('validating', {})
    if validating.get('full_validation', False) is not False or validating.get('ema_full_validation', False) is not False:
        raise ValueError('unsupported full or EMA validation option')


def verify_resume_lr(saved_groups, current_groups):
    if len(saved_groups) != len(current_groups):
        raise ValueError('optimizer group count mismatch')
    for before, after in zip(saved_groups, current_groups):
        x, y = float(before['lr']), float(after['lr'])
        if not math.isfinite(x) or not math.isfinite(y) or not math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-15):
            raise ValueError('restored optimizer learning rate differs from checkpoint')


def finite_step(original, trainer, torch):
    """Fail before update on nonfinite gradients, and after update on parameters."""
    params = [p for group in trainer.optimizer.param_groups for p in group['params']]
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise ValueError('optimizer has no gradients')
    if any(g.is_sparse for g in grads):
        raise ValueError('sparse gradients are unsupported')
    def finite_tensors(tensors):
        groups = {}
        for tensor in tensors:
            if tensor.is_floating_point() or tensor.is_complex():
                groups.setdefault(str(tensor.device), []).append(torch.isfinite(tensor).all())
        return all(bool(torch.stack(values).all().item()) for values in groups.values())
    if any(not math.isfinite(float(g['lr'])) for g in trainer.optimizer.param_groups):
        raise FloatingPointError('nonfinite learning rate')
    if not finite_tensors(grads):
        raise FloatingPointError('nonfinite gradient')
    original(trainer)
    if not finite_tensors(params):
        raise FloatingPointError('nonfinite parameter after optimizer step')
    def leaves(value):
        if isinstance(value, torch.Tensor): yield value
        elif isinstance(value, dict):
            for x in value.values(): yield from leaves(x)
        elif isinstance(value, (list, tuple)):
            for x in value: yield from leaves(x)
    if not finite_tensors(leaves(trainer.optimizer.state)):
        raise FloatingPointError('nonfinite optimizer state after optimizer step')
    if any(not math.isfinite(float(g['lr'])) for g in trainer.optimizer.param_groups):
        raise FloatingPointError('nonfinite learning rate after optimizer step')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('start', 'resume'))
    for key in ('config', 'run', 'train', 'val', 'initial', 'initial-sha256'):
        p.add_argument('--'+key, required=True)
    args = p.parse_args()
    if os.environ.get('WORLD_SIZE', '1') != '1':
        raise ValueError('single rank only')
    config, run, train, val, initial = [Path(getattr(args, x)).resolve(strict=True) for x in ('config', 'run', 'train', 'val', 'initial')]
    if not _HASH.fullmatch(args.initial_sha256) or _sha(initial) != args.initial_sha256:
        raise ValueError('initial checkpoint hash mismatch')
    config_bytes = config.read_bytes()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    cfg = json.loads(config_bytes)
    validate_config(cfg, train, val)
    import numpy as np
    import torch
    import deepmd
    from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem
    from deepmd.pt_expt.train.training import Trainer
    import deepmd.pt_expt.utils.lmdb_dataset as dataset_module
    import deepmd.dpmodel.utils.lmdb_data as reader_module
    import deepmd.pt_expt.train.training as training_module
    import deepmd.dpmodel.train.checkpoint as checkpoint_module
    import deepmd.dpmodel.train.trainer as trainer_module
    import deepmd.pt.optimizer.hybrid_muon as optimizer_module
    import deepmd.dpmodel.utils.learning_rate as lr_module
    modules = (dataset_module, reader_module, training_module, checkpoint_module,
               trainer_module, optimizer_module, lr_module)
    sources = {str(Path(m.__file__).resolve()): _sha(m.__file__) for m in modules}
    expected_sources = {
        Path(dataset_module.__file__).resolve(): 'ea87d6bd521d23dda2cc047808bc6495a18462f24c13ef70e9cf44761ab738e4',
        Path(reader_module.__file__).resolve(): '9c7c086335ab5bc1e0dbb8bea6482756e66196468796abd799693f045d4840f0',
        Path(training_module.__file__).resolve(): 'f804b39208f7101ccfddff0803f462e6dfbf8667f9b295fdda3d0f2531b4277e',
        Path(checkpoint_module.__file__).resolve(): '4cc40c5f7e270faf3d201c1bbd5f06bfd01d6adb02bbc4b04ea7e6b4f46a8ef8',
        Path(trainer_module.__file__).resolve(): 'd9f81b5f16d42cbe27ec7576bf70409663532c1d30d4e8e8ce8287b82eff4793',
        Path(optimizer_module.__file__).resolve(): 'ea4cda9a186c09365a0c860c0548c2c25084b126753b312065ed51586cb8c279',
        Path(lr_module.__file__).resolve(): '23cde8949987dd3da3982027e8697682a473a74cf1e20227975c5ea1fc9ccd47',
    }
    if any(sources[str(path)] != expected for path, expected in expected_sources.items()):
        raise ValueError('unsupported DeepMD source build')
    local_sources = {name: _sha(Path(__file__).parent/name) for name in
                     ('restricted_train_batch8_experimental.py', 'sampler_resume_experimental.py', 'sampler_recovery.py')}
    dp = Path(shutil.which('dp') or '').resolve(strict=True)
    info = {'config_path': str(config), 'config_sha256': config_digest,
            'train_path': str(train), 'train_sha256': _sha(train/'data.mdb'),
            'val_path': str(val), 'val_sha256': _sha(val/'data.mdb'),
            'initial_path': str(initial), 'initial_sha256': args.initial_sha256,
            'deepmd_sources': sources, 'local_sources': local_sources, 'dp_sha256': _sha(dp), 'python': sys.version,
            'deepmd': deepmd.__version__, 'torch': torch.__version__, 'numpy': np.__version__,
            'world_size': 1, 'workers': os.environ.get('DP_LMDB_NUM_WORKERS'),
            'rng_state_saved': False}
    ident = fingerprint(info)
    meta = run / 'restricted_identity.json'
    fd = os.open(run / '.restricted.lock', os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_run_symlinks(run)
        if (run/'restricted_complete.json').exists():
            raise ValueError('run is already marked complete')
        snapshot = run / 'restricted_input.json'
        if args.mode == 'start':
            if {p.name for p in run.iterdir()} != {'.restricted.lock'}:
                raise FileExistsError('new run directory must be empty except for lock')
            with snapshot.open('xb') as f:
                f.write(config_bytes); f.flush(); os.fsync(f.fileno())
            snapshot.chmod(0o400)
            with meta.open('x') as f:
                json.dump({'fingerprint':ident, 'identity':info},f,sort_keys=True,indent=2); f.flush(); os.fsync(f.fileno())
            _fsync_dir(run)
            checkpoint = None
        else:
            if snapshot.is_symlink() or _sha(snapshot) != config_digest:
                raise ValueError('run configuration snapshot changed')
            if json.loads(meta.read_text()) != {'fingerprint':ident, 'identity':info}:
                raise ValueError('run identity mismatch')
            pairs = inspect_pairs(run, ident)
            if not pairs['safe_to_launch']:
                raise ValueError('no complete conflict-free checkpoint pair: '+json.dumps(pairs))
            checkpoint = Path(pairs['selected']['checkpoint'])
            if pairs['selected']['step'] >= cfg['training']['num_steps']:
                raise ValueError('checkpoint already reached configured total steps')
        state = DurableSamplerResume(train, val, ident, checkpoint)
        install(LmdbDataSystem, Trainer, state)
        old_write = Trainer._write_checkpoint
        def checked_write(trainer, path, **kw):
            path = Path(path)
            if path.resolve().parent != run:
                raise ValueError('checkpoint outside locked run directory')
            if path.is_symlink() or Path(str(path)+'.sampler.json').is_symlink():
                raise FileExistsError('refusing checkpoint or sidecar symlink')
            return old_write(trainer, path, **kw)
        Trainer._write_checkpoint = checked_write
        old_run, old_step, old_train_step = Trainer.run, Trainer._optimizer_step, Trainer.train_step
        def checked_run(trainer, *a, **kw):
            if checkpoint:
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                verify_resume_lr(saved['optimizer']['param_groups'], trainer.optimizer.state_dict()['param_groups'])
            return old_run(trainer, *a, **kw)
        Trainer.run = checked_run
        Trainer._optimizer_step = lambda trainer: finite_step(old_step, trainer, torch)
        def checked_train_step(trainer, *a, **kw):
            result = old_train_step(trainer, *a, **kw)
            if not bool(torch.isfinite(torch.as_tensor(result.payload['loss'])).all().item()):
                raise FloatingPointError('nonfinite loss after train step')
            return result
        Trainer.train_step = checked_train_step
        print('RESTRICTED_EXPERIMENTAL identity='+ident+' rng_state_saved=false', flush=True)
        os.chdir(run)
        sys.argv = [str(dp), '--pt-expt', 'train', str(snapshot)]
        sys.argv += ['--restart', str(checkpoint)] if checkpoint else ['--init-model', str(initial)]
        try:
            runpy.run_path(str(dp), run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        if _sha(snapshot) != config_digest or _sha(train/'data.mdb') != info['train_sha256'] or _sha(val/'data.mdb') != info['val_sha256']:
            raise ValueError('configuration or data changed during training')
        pairs = inspect_pairs(run, ident)
        if not pairs['safe_to_launch'] or pairs['selected']['step'] != cfg['training']['num_steps']:
            raise RuntimeError('training returned without a valid final checkpoint pair')
        with (run/'restricted_complete.json').open('x') as f:
            json.dump({'status':'complete', 'fingerprint':ident, 'checkpoint':pairs['selected'], 'rng_state_saved':False},f,indent=2)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        _fsync_dir(run)


if __name__ == '__main__':
    main()
