"""Prepare an unexecuted DPA4C-only research configuration for complete LMDBs.

This does not certify dataset quality, GPU compatibility, model accuracy,
training memory bounds or train-only fine-tuning bias statistics. The caller
must evaluate retained checkpoints on the full val split and leave test held
out. For actual training the released checkpoint must be supplied separately.
"""
import argparse
import json
from pathlib import Path

from cace_sog_training.data import IndexReader
from .index_smoke import _identity, _release
from .prepare import sha256


def prepare(database, release_config, converted, output, steps, batch_size, seed):
    for name, value, minimum in (('steps', steps, 1), ('batch_size', batch_size, 1), ('seed', seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    data = Path(converted).expanduser().resolve(strict=True)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError('configuration output already exists')
    if target == data or data in target.parents:
        raise ValueError('configuration output must be outside the converted directory')
    source = Path(release_config).expanduser().resolve(strict=True)
    config, release_hash, _ = _release(source)
    status_path = data / 'status.json'
    status_hash = sha256(status_path)
    status = json.loads(status_path.read_text(encoding='utf8'))
    if (status.get('status') != 'complete' or status.get('channel') != 'sog_candidates'
            or status.get('debug_only') is not False or status.get('no_sampling') is not True
            or status.get('test_split_read') is not False
            or status.get('condition') != [0, 1]
            or status.get('condition_is_not_measured_dft_state') is not True):
        raise ValueError('requires complete, unsampled train/val conversion with recorded condition')
    with IndexReader(database) as reader:
        identity = _identity(reader.metadata)
        counts = {s: reader.counts('sog_candidates', s) for s in ('train', 'val')}
        if identity != status.get('source_index_metadata_sha256'):
            raise ValueError('source index identity differs from conversion')
        if counts != status.get('counts') or counts != status.get('source_counts'):
            raise ValueError('index and converted split counts differ')
        reader.verify_sources()
    hashes = {}
    for split in ('train', 'val'):
        for kind, rel in (('lmdb', f'{split}.lmdb/data.mdb'), ('manifest', f'{split}.jsonl')):
            path = data / rel
            before = path.stat()
            actual = sha256(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'{rel} changed while hashing')
            if actual != status.get(f'{split}_{kind}_sha256'):
                raise ValueError(f'{rel} SHA-256 differs from conversion status')
            hashes[f'{split}_{kind}_sha256'] = actual
    if sha256(status_path) != status_hash:
        raise ValueError('conversion status changed during checks')
    training = config.setdefault('training', {})
    for key in ('stat_file', 'stat_file_mode', 'numb_epoch', 'numb_epochs', 'num_epoch', 'num_epochs',
                'numb_step', 'numb_steps', 'num_step', 'num_steps', 'stop_batch', 'zero_stage'):
        training.pop(key, None)
    training.update({'training_data': {'systems': str(data / 'train.lmdb'), 'batch_size': batch_size},
                     'validation_data': {'systems': str(data / 'val.lmdb'), 'batch_size': batch_size, 'numb_batch': 1},
                     'num_steps': steps, 'seed': seed, 'enable_compile': False, 'enable_ema': False})
    # Always emit at least one checkpoint even for a short requested run.
    training['save_freq'] = min(2000, steps)
    training.setdefault('max_ckpt_keep', 3)
    learning = config.setdefault('learning_rate', {})
    learning['start_lr'] = 1e-4
    if 'stop_lr' in learning:
        learning['stop_lr'] = min(float(learning['stop_lr']), 1e-5)
    config['validating'] = {'full_validation': False, 'ema_full_validation': False}
    target.mkdir(parents=True)
    config_path = target / 'input_dpa4c_lmdb.json'
    config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + '\n', encoding='utf8')
    report = {'status': 'prepared_only', 'no_training_executed': True, 'backend': 'pt_expt',
              'test_split_read': False, 'debug_only': False, 'source_index_identity': identity,
              'source_counts': counts, 'release_config_sha256': release_hash,
              'conversion_status_sha256': status_hash, 'converted_hashes': hashes,
              'config_sha256': sha256(config_path), 'condition': [0, 1],
              'condition_is_not_measured_dft_state': True,
              'sampling': 'one LMDB logical system per split; fixed steps do not imply full epoch',
              'bias_note': 'Fine-tune startup statistics must be train-only; verify LMDB statistics and elemental coverage before use.',
              'validation_note': 'Periodic one-batch validation is diagnostic only; evaluate all val frames before choosing a checkpoint.',
              'limitations': 'Preparation does not verify SCNet LMDB training, restart, memory, accuracy or readiness.'}
    (target / 'preparation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('database', 'release-config', 'converted', 'output'):
        parser.add_argument('--' + key, required=True)
    for key in ('steps', 'batch-size', 'seed'):
        parser.add_argument('--' + key, required=True, type=int)
    a = parser.parse_args()
    print(json.dumps(prepare(a.database, a.release_config, a.converted, a.output,
                             a.steps, a.batch_size, a.seed), indent=2))


if __name__ == '__main__':
    main()
