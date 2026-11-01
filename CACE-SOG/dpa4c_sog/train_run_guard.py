"""Verify prepared full LMDB inputs and guard DPA4C training provenance.

This guard is not a numerical, physical, memory or training-readiness test.
"""
import argparse
import json
import re
from pathlib import Path

from cace_sog_training.data import IndexReader
from .index_smoke import _identity, _release
from .prepare import sha256

PRETRAINED_SHA256 = 'fc42df87b4c81649755e33c64d7e95bc0584a353e01c8f313742b3f73aab412c'
RELEASE_SHA256 = '96a92146afe26ccb5f9a6687bfa8bfd9013f3b1ff654fb40c949e220fa802e31'


def checked_files(prepared, database, release, pretrained, repo_commit):
    prepared = Path(prepared).resolve(strict=True)
    config_path = prepared / 'input_dpa4c_lmdb.json'
    report_path = prepared / 'preparation.json'
    report = json.loads(report_path.read_text(encoding='utf8'))
    config = json.loads(config_path.read_text(encoding='utf8'))
    if (report.get('status') != 'prepared_only' or report.get('debug_only') is not False
            or report.get('no_training_executed') is not True or report.get('test_split_read') is not False
            or report.get('backend') != 'pt_expt' or report.get('condition') != [0, 1]
            or report.get('condition_is_not_measured_dft_state') is not True):
        raise ValueError('expected full non-debug DPA4C preparation')
    if sha256(config_path) != report.get('config_sha256'):
        raise ValueError('prepared configuration SHA mismatch')
    if sha256(pretrained) != PRETRAINED_SHA256:
        raise ValueError('released pretrained checkpoint SHA mismatch')
    released, release_hash, _ = _release(Path(release).resolve(strict=True))
    if release_hash != RELEASE_SHA256 or report.get('release_config_sha256') != release_hash:
        raise ValueError('released configuration SHA mismatch')
    if config.get('model') != released.get('model'):
        raise ValueError('prepared model differs from released DPA4C model')
    training = config['training']
    if training.get('enable_compile') is not False:
        raise ValueError('compile must remain disabled')
    train_path = Path(training['training_data']['systems']).resolve(strict=True)
    val_path = Path(training['validation_data']['systems']).resolve(strict=True)
    if train_path.name != 'train.lmdb' or val_path.name != 'val.lmdb' or train_path.parent != val_path.parent:
        raise ValueError('expected train/val LMDBs under the same conversion directory')
    data = train_path.parent
    status_path = data / 'status.json'
    if sha256(status_path) != report.get('conversion_status_sha256'):
        raise ValueError('conversion status SHA mismatch')
    status = json.loads(status_path.read_text(encoding='utf8'))
    if (status.get('status') != 'complete' or status.get('debug_only') is not False
            or status.get('no_sampling') is not True or status.get('test_split_read') is not False
            or status.get('counts') != report.get('source_counts')):
        raise ValueError('converted data are incomplete or sampled')
    with IndexReader(database) as reader:
        identity = _identity(reader.metadata)
        if identity != report.get('source_index_identity'):
            raise ValueError('bound index identity mismatch')
        if {s: reader.counts('sog_candidates', s) for s in ('train', 'val')} != report['source_counts']:
            raise ValueError('index frame counts mismatch')
    for split in ('train', 'val'):
        for kind, file in (('lmdb', data / (split + '.lmdb') / 'data.mdb'),
                           ('manifest', data / (split + '.jsonl'))):
            key = split + '_' + kind + '_sha256'
            before = file.stat(); digest = sha256(file); after = file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('file changed during hashing: ' + str(file))
            if digest != status.get(key) or digest != report.get('converted_hashes', {}).get(key):
                raise ValueError('converted data hash mismatch: ' + str(file))
    if not re.fullmatch('[0-9a-f]{40}', repo_commit):
        raise ValueError('repository commit must be a complete SHA-1')
    return {'version': 1, 'prepared_dir': str(prepared), 'database': str(Path(database).resolve()),
            'release': str(Path(release).resolve()), 'pretrained': str(Path(pretrained).resolve()),
            'repository_commit': repo_commit, 'preparation_sha256': sha256(report_path),
            'config_sha256': sha256(config_path), 'pretrained_sha256': PRETRAINED_SHA256,
            'source_index_identity': identity, 'source_counts': report['source_counts'],
            'conversion_status_sha256': report['conversion_status_sha256'],
            'converted_hashes': report['converted_hashes'], 'test_split_read': False,
            'input_assumption': '[0,1] is a model input, not a measured electronic state'}


def start(run, checked):
    run = Path(run).resolve(strict=True)
    provenance = run / 'run_provenance.json'
    local_config = run / 'input_dpa4c_lmdb.json'
    if provenance.exists() or local_config.exists():
        raise FileExistsError('run provenance or configuration already exists')
    source = Path(checked['prepared_dir']) / 'input_dpa4c_lmdb.json'
    with local_config.open('xb') as f:
        f.write(source.read_bytes()); f.flush()
    if sha256(local_config) != checked['config_sha256']:
        raise ValueError('copied configuration SHA mismatch')
    with provenance.open('x', encoding='utf8') as f:
        json.dump(checked, f, indent=2, allow_nan=False); f.write('\n')


def resume(run, checked, checkpoint, expected_sha256):
    run = Path(run).resolve(strict=True)
    if (run / 'training_status.json').exists():
        raise ValueError('this run already has a completed training status')
    if json.loads((run / 'run_provenance.json').read_text()) != checked:
        raise ValueError('run provenance differs from current inputs or code commit')
    if sha256(run / 'input_dpa4c_lmdb.json') != checked['config_sha256']:
        raise ValueError('run configuration changed')
    checkpoint = Path(checkpoint).resolve(strict=True)
    if checkpoint.parent != run or not re.fullmatch(r'model\.ckpt-\d+\.pt', checkpoint.name):
        raise ValueError('restart checkpoint must be a numbered checkpoint in this run')
    candidates = [(int(m.group(1)), p) for p in run.glob('model.ckpt-*.pt')
                  if (m := re.fullmatch(r'model\.ckpt-(\d+)\.pt', p.name))]
    if not candidates or checkpoint != max(candidates, key=lambda x: x[0])[1].resolve():
        raise ValueError('restart must use the latest numbered checkpoint to preserve newer files')
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256) or sha256(checkpoint) != expected_sha256:
        raise ValueError('restart checkpoint SHA mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('start', 'resume'), required=True)
    for name in ('prepared', 'database', 'release', 'pretrained', 'repo-commit', 'run'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--checkpoint-sha256')
    a = p.parse_args()
    checked = checked_files(a.prepared, a.database, a.release, a.pretrained, a.repo_commit)
    if a.mode == 'start':
        if a.checkpoint or a.checkpoint_sha256:
            raise ValueError('start mode must not specify a restart checkpoint')
        start(a.run, checked)
    else:
        if not a.checkpoint or not a.checkpoint_sha256:
            raise ValueError('resume mode requires checkpoint and its SHA-256')
        resume(a.run, checked, a.checkpoint, a.checkpoint_sha256)
    print('DPA4C_TRAIN_PROVENANCE_CHECK_OK', flush=True)


if __name__ == '__main__':
    main()
