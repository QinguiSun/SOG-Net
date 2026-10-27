"""Prepare research configurations from the matching released DPA4C JSON.

This creates inputs; it does not install DeePMD, run training, or certify the
upstream experimental periodic-SOG implementation. The train/val paths point
to a complete split-preserving conversion, and test is never a training input.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

from ase.data import chemical_symbols


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(release_config, converted, output, train_steps, sog_steps,
            batch_size=1, start_lr=1e-4, seed=42, save_every=10000, keep_checkpoints=5,
            integrity_report=None, alignment_report=None):
    for name, val in (("train_steps", train_steps), ("sog_steps", sog_steps), ("batch_size", batch_size),
                      ("save_every", save_every), ("keep_checkpoints", keep_checkpoints)):
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError(name + " must be a positive integer")
    if not math.isfinite(start_lr) or start_lr <= 0:
        raise ValueError("start_lr must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    source = Path(release_config).expanduser().resolve(strict=True)
    data = Path(converted).expanduser().resolve(strict=True)
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError("output directory exists; choose a new directory")
    config = json.loads(source.read_text(encoding='utf-8-sig'))
    status_file = data / 'conversion_status.json'
    status = json.loads(status_file.read_text(encoding='utf-8-sig'))
    if status.get('status') != 'complete' or status.get('channel') != 'sog_candidates':
        raise ValueError('requires a complete sog_candidates conversion')
    if status.get('assumed_charge_spin') != [0, 1] or not status.get('charge_spin_is_model_input_assumption_not_verified_dft_state'):
        raise ValueError('conversion must record the explicit [0,1] input assumption')
    if integrity_report is None or alignment_report is None:
        raise ValueError('complete integrity and source-alignment reports are required')
    integrity_path = Path(integrity_report).expanduser().resolve(strict=True)
    alignment_path = Path(alignment_report).expanduser().resolve(strict=True)
    integrity = json.loads(integrity_path.read_text(encoding='utf-8-sig'))
    alignment = json.loads(alignment_path.read_text(encoding='utf-8-sig'))
    status_digest = sha256(status_file)
    if (integrity.get('status') != 'complete' or
            Path(integrity.get('target_dataset', '')).resolve() != data or
            integrity.get('conversion_status_sha256') != status_digest):
        raise ValueError('integrity report does not match this complete target dataset')
    if (alignment.get('status') != 'complete' or alignment.get('channel') != 'sog_candidates' or
            Path(alignment.get('converted_dir', '')).resolve() != data or
            alignment.get('conversion_status_sha256') != status_digest or
            alignment.get('counts') != status.get('counts') or
            alignment.get('frames_manifest_sha256') != sha256(data / 'frames.jsonl')):
        raise ValueError('alignment report does not match this conversion and frame manifest')
    if any(not (data / split).is_dir() or status.get('counts', {}).get(split, 0) < 1
           for split in ('train', 'val', 'test')):
        raise ValueError('conversion must contain nonempty train, val and test splits')
    model = config.get('model', {})
    descriptor = model.get('descriptor', {})
    if model.get('type_map') != chemical_symbols[1:119] or descriptor.get('type') != 'dpa4c':
        raise ValueError('release must contain the full H–Og DPA4C model section')
    if descriptor.get('add_chg_spin_ebd') is not True or descriptor.get('default_chg_spin') != [0, 1]:
        raise ValueError('release must use the reviewed default charge/spin conditioning [0,1]')
    if 'long_range' in model:
        raise ValueError('release baseline must not already include a long-range module')
    baseline = copy.deepcopy(config)
    training = baseline.setdefault('training', {})
    for key in ('stat_file', 'stat_file_mode', 'numb_epoch', 'numb_epochs', 'num_epoch', 'num_epochs',
                'numb_step', 'numb_steps', 'num_step', 'num_steps', 'stop_batch', 'zero_stage'):
        training.pop(key, None)
    training.update({'training_data': {'systems': str(data / 'train'), 'batch_size': batch_size,
                                       'auto_prob_style': 'prob_sys_size'},
                     'validation_data': {'systems': str(data / 'val'), 'batch_size': batch_size, 'numb_batch': 1},
                     'num_steps': train_steps, 'seed': seed, 'enable_compile': False,
                     'save_freq': min(save_every, train_steps), 'max_ckpt_keep': keep_checkpoints,
                     'enable_ema': False})
    learning = baseline.setdefault('learning_rate', {})
    learning['start_lr'] = start_lr
    if 'stop_lr' in learning:
        learning['stop_lr'] = min(float(learning['stop_lr']), start_lr / 10)
    # Upstream FullValidator currently materializes and retains each complete
    # validation system. Select saved checkpoints using our streaming validator.
    baseline['validating'] = {'full_validation': False, 'ema_full_validation': False}
    sog = copy.deepcopy(baseline)
    sog['training']['num_steps'] = sog_steps
    sog['training']['save_freq'] = min(save_every, sog_steps)
    sog['model']['long_range'] = {'type': 'latent_charge', 'channels': 1,
        'kernel': {'type': 'sog'}, 'periodic': True, 'k_grid_mode': 'pme',
        'charge_constraint': {'source': 'neutral'}}
    target.mkdir(parents=True)
    for filename, value in (('input_dpa4c.json', baseline), ('input_sog_experimental.json', sog)):
        (target / filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    provenance = {'status': 'prepared_only', 'channel': 'sog_candidates', 'no_training_executed': True,
        'release_config_sha256': sha256(source), 'conversion_status_sha256': status_digest,
        'integrity_report_sha256': sha256(integrity_path), 'alignment_report_sha256': sha256(alignment_path),
        'bound_index_identity': alignment.get('bound_index_identity'),
        'bound_database': alignment.get('bound_database'),
        'source_index_metadata_sha256': status.get('source_index_metadata_sha256'),
        'split_counts': status['counts'], 'assumed_charge_spin': [0, 1],
        'charge_spin_policy': 'model input assumption, not a verified physical electronic state',
        'sog_status': 'experimental upstream PR; periodic validation is not established',
        'upstream_pr_head_reviewed': 'ceefa8f806fa6e8607d79d10d8316b4ab325ad11',
        'validation_policy': 'upstream full validation disabled to avoid unbounded caching; stream full val for each retained checkpoint and choose force-component MAE',
        'sampling_warning': 'prob_sys_size samples systems by batch counts; a fixed number of optimizer steps does not guarantee every frame is seen',
        'train_steps': train_steps, 'sog_steps': sog_steps, 'batch_size': batch_size, 'start_lr': start_lr, 'seed': seed,
        'save_every': save_every, 'keep_checkpoints': keep_checkpoints}
    (target / 'preparation.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-config', required=True, type=Path)
    parser.add_argument('--converted', required=True, type=Path)
    parser.add_argument('--integrity-report', required=True, type=Path)
    parser.add_argument('--alignment-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--train-steps', required=True, type=int)
    parser.add_argument('--sog-steps', required=True, type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--start-lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int, default=10000)
    parser.add_argument('--keep-checkpoints', type=int, default=5)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.release_config, args.converted, args.output, args.train_steps, args.sog_steps,
                         args.batch_size, args.start_lr, args.seed, args.save_every, args.keep_checkpoints,
                         args.integrity_report, args.alignment_report)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
