import copy
import math
from pathlib import Path

import pytest

from dpa4c_sog.restricted_train_experimental import fingerprint, validate_config, verify_resume_lr


def config(tmp_path):
    train, val = tmp_path / 'train.lmdb', tmp_path / 'val.lmdb'
    cfg = {'model': {'descriptor': {'type': 'dpa4c'}}, 'training': {
        'training_data': {'systems': str(train), 'batch_size': 1, 'auto_prob_style': 'prob_sys_size'},
        'validation_data': {'systems': str(val), 'batch_size': 1, 'numb_batch': 1},
        'enable_compile': False, 'enable_ema': False, 'seed': 42,
        'num_steps': 16, 'save_freq': 4, 'max_ckpt_keep': 3}}
    return cfg, train, val


def test_identity_is_order_independent_and_binds_inputs():
    a = {'config': 'a', 'train': 'b', 'version': '3.2'}
    assert fingerprint(a) == fingerprint(dict(reversed(list(a.items()))))
    assert fingerprint(a) != fingerprint(dict(a, train='different'))


def test_config_rejects_wrong_input_and_topology(tmp_path):
    cfg, train, val = config(tmp_path)
    validate_config(cfg, train, val)
    bad = copy.deepcopy(cfg); bad['training']['training_data']['batch_size'] = 8
    with pytest.raises(ValueError): validate_config(bad, train, val)
    bad = copy.deepcopy(cfg); bad['training']['enable_ema'] = True
    with pytest.raises(ValueError): validate_config(bad, train, val)
    bad = copy.deepcopy(cfg); bad['training']['max_ckpt_keep'] = 1
    with pytest.raises(ValueError): validate_config(bad, train, val)
    for key, value in (('zero_stage', 1), ('change_bias_after_training', True), ('ckpt_keep_ratio', 0.5), ('seed', 43),
                       ('save_dir', '../outside'), ('save_ckpt', '/tmp/other'), ('disp_file', '../outside'),
                       ('profiling', True), ('tensorboard', True)):
        bad = copy.deepcopy(cfg); bad['training'][key] = value
        with pytest.raises(ValueError): validate_config(bad, train, val)
    with pytest.raises(ValueError): validate_config(cfg, val, train)


def test_lr_check_rejects_drift_and_nonfinite():
    verify_resume_lr([{'lr': 1e-4}], [{'lr': 1e-4}])
    for bad in (0.1, math.nan, math.inf):
        with pytest.raises(ValueError): verify_resume_lr([{'lr': 1e-4}], [{'lr': bad}])
    with pytest.raises(ValueError): verify_resume_lr([{'lr': 1e-4}], [])
