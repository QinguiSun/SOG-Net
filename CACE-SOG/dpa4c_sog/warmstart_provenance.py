"""Read-only guard for the interrupted train-only batch-8 warm start.

This verifies provenance and the checkpoint/sidecar pairing, not scientific
accuracy or numerical state. A new --init-model run resets optimizer, scheduler,
sampler and step; it is not a continuation of the old run.
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_ID = 'c07665508e5499fa61dcb5d2faa06ef81fdce40da68f0a88a308ecaf24795251'
PILOT_SHA = 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355'
TRAIN_SHA = '04801d713893efdeef0dac477f38d7ae613565999fbe483224eff7f50f9842b3'
VAL_SHA = '634d37b82114017e300fc4e32517ce3574287312a758b0f4bdd7b985a25a4b92'
B8_CODE_SHA = '471ef0b5d96274f4dda5f30b3f6b30f00ddba8c920bbc3f42fea33c1115e38de'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_file(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError('expected a regular non-symlink file: ' + str(p))
    before = p.stat(); value = p.read_bytes(); after = p.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed while reading: ' + str(p))
    return value


def stable_digest(path):
    p = Path(path)
    if p.is_symlink() or not p.is_file(): raise ValueError('expected a regular non-symlink file: ' + str(p))
    before = p.stat(); value = digest(p); after = p.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed while hashing: ' + str(p))
    return value


def verify(run, prepared, conversion, expected_checkpoint_sha, expected_fingerprint):
    run, prepared, conversion = map(lambda p: Path(p).resolve(strict=True), (run, prepared, conversion))
    cp = run / 'model.ckpt-245000.pt'
    side = json.loads(read_file(Path(str(cp) + '.sampler.json')))
    identity = json.loads(read_file(run / 'restricted_identity.json'))
    info = identity['identity']
    actual_identity = hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if identity['fingerprint'] != actual_identity or actual_identity != expected_fingerprint:
        raise ValueError('old run identity fingerprint mismatch')
    if cp.is_symlink() or not cp.is_file(): raise ValueError('invalid checkpoint')
    before = cp.stat(); cp_sha = digest(cp); after = cp.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('checkpoint changed during hashing')
    if cp_sha != expected_checkpoint_sha or side.get('checkpoint_sha256') != cp_sha:
        raise ValueError('checkpoint SHA mismatch')
    if side.get('fingerprint') != actual_identity or side.get('step') != 245000 or side.get('version') != 1:
        raise ValueError('checkpoint sidecar identity/step mismatch')
    if side.get('batch_counts', {}).get('train') != 245000 or side.get('seeds', {}).get('train') != 42:
        raise ValueError('unexpected training sampler progress')
    config_bytes = read_file(run / 'restricted_input.json')
    if hashlib.sha256(config_bytes).hexdigest() != info['config_sha256']:
        raise ValueError('old run config snapshot mismatch')
    cfg = json.loads(config_bytes); t = cfg['training']
    if (t['training_data']['batch_size'] != 8 or t['training_data'].get('auto_prob_style') != 'prob_sys_size'
            or t['validation_data']['batch_size'] != 1 or t['seed'] != 42 or t['num_steps'] != 442978
            or t.get('change_bias_after_training') is not False):
        raise ValueError('old run configuration mismatch')
    if Path(t['training_data']['systems']).resolve() != (conversion / 'train.lmdb').resolve() or Path(t['validation_data']['systems']).resolve() != (conversion / 'val.lmdb').resolve():
        raise ValueError('old run dataset paths mismatch')
    if (Path(info['train_path']).resolve() != (conversion / 'train.lmdb').resolve()
            or Path(info['val_path']).resolve() != (conversion / 'val.lmdb').resolve()
            or Path(info['config_path']).resolve() != (prepared / 'input_dpa4c_lmdb.json').resolve()):
        raise ValueError('old run identity paths mismatch')
    if (info.get('train_sha256') != TRAIN_SHA or info.get('val_sha256') != VAL_SHA
            or info.get('initial_sha256') != PILOT_SHA
            or info.get('local_sources', {}).get('restricted_train_batch8_experimental.py') != B8_CODE_SHA):
        raise ValueError('old run source identity mismatch')
    prep = json.loads(read_file(prepared / 'preparation.json'))
    if (prep.get('source_index_identity') != SOURCE_ID or prep.get('debug_only') is not False
            or prep.get('no_training_executed') is not True or prep.get('test_split_read') is not False
            or prep.get('source_counts') != {'train': 3543786, 'val': 443734}
            or prep.get('config_sha256') != info['config_sha256']):
        raise ValueError('old prepared dataset identity mismatch')
    status = json.loads(read_file(conversion / 'status.json'))
    if (status.get('status') != 'complete' or status.get('no_sampling') is not True
            or status.get('test_split_read') is not False or status.get('source_index_metadata_sha256') != SOURCE_ID
            or status.get('counts') != {'train': 3543786, 'val': 443734}
            or status.get('train_lmdb_sha256') != TRAIN_SHA or status.get('val_lmdb_sha256') != VAL_SHA):
        raise ValueError('conversion status mismatch')
    if stable_digest(conversion / 'train.lmdb' / 'data.mdb') != TRAIN_SHA or stable_digest(conversion / 'val.lmdb' / 'data.mdb') != VAL_SHA:
        raise ValueError('converted train/val hashes mismatch')
    return {'status': 'verified', 'checkpoint': str(cp), 'checkpoint_sha256': cp_sha,
            'old_run_fingerprint': actual_identity, 'old_checkpoint_step': 245000,
            'source_index_identity': SOURCE_ID, 'test_split_read': False,
            'new_stage': 'init-model: weights/bias retained; optimizer/scheduler/step/sampler reset',
            'requires_validation': 'actual DeepMD checkpoint deserialization and prediction are separate checks'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'prepared', 'conversion', 'checkpoint-sha256', 'fingerprint'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.run, a.prepared, a.conversion, a.checkpoint_sha256, a.fingerprint), indent=2))


if __name__ == '__main__': main()
