import json
from pathlib import Path

import pytest
from ase.data import chemical_symbols

from dpa4c_sog.prepare import prepare, sha256


def inputs(tmp_path):
    release = {'model': {'type_map': chemical_symbols[1:119],
        'descriptor': {'type': 'dpa4c', 'add_chg_spin_ebd': True,
                       'default_chg_spin': [0, 1], 'rcut': 6},
        'fitting_net': {'neuron': [256, 256, 256]}},
        'training': {'stat_file': 'old-OMol.hdf5', 'num_epochs': 25, 'zero_stage': 1},
        'learning_rate': {'type': 'wsd', 'start_lr': 0.003, 'stop_lr': 1e-6}}
    release_path = tmp_path / 'release.json'
    release_path.write_text(json.dumps(release))
    data = tmp_path / 'converted'
    for name in ('train', 'val', 'test'):
        (data / name).mkdir(parents=True)
    (data / 'conversion_status.json').write_text(json.dumps({'status': 'complete',
        'channel': 'sog_candidates', 'assumed_charge_spin': [0, 1],
        'charge_spin_is_model_input_assumption_not_verified_dft_state': True,
        'counts': {'train': 5, 'val': 2, 'test': 3}}))
    (data / 'frames.jsonl').write_text('{}\n')
    status_hash = sha256(data / 'conversion_status.json')
    integrity = tmp_path / 'integrity.json'
    alignment = tmp_path / 'alignment.json'
    integrity.write_text(json.dumps({'status': 'complete', 'target_dataset': str(data),
                                     'conversion_status_sha256': status_hash}))
    alignment.write_text(json.dumps({'status': 'complete', 'channel': 'sog_candidates',
        'converted_dir': str(data), 'conversion_status_sha256': status_hash,
        'frames_manifest_sha256': sha256(data / 'frames.jsonl'),
        'counts': {'train': 5, 'val': 2, 'test': 3}}))
    return release_path, data, release, integrity, alignment


def test_prepare_preserves_backbone_and_never_uses_test_for_training(tmp_path):
    release, data, original, integrity, alignment = inputs(tmp_path)
    output = tmp_path / 'inputs'
    report = prepare(release, data, output, train_steps=20, sog_steps=10, batch_size=2,
                     integrity_report=integrity, alignment_report=alignment)
    baseline = json.loads((output / 'input_dpa4c.json').read_text())
    sog = json.loads((output / 'input_sog_experimental.json').read_text())
    assert baseline['model'] == original['model']
    assert {k: v for k, v in sog['model'].items() if k != 'long_range'} == original['model']
    assert sog['model']['long_range']['kernel']['type'] == 'sog'
    assert sog['model']['long_range']['periodic'] is True
    assert baseline['training']['training_data']['systems'] == str(data / 'train')
    assert baseline['training']['validation_data']['systems'] == str(data / 'val')
    assert str(data / 'test') not in json.dumps(baseline['training'])
    assert 'num_epochs' not in baseline['training'] and 'stat_file' not in baseline['training']
    assert baseline['training']['num_steps'] == 20 and sog['training']['num_steps'] == 10
    assert baseline['validating']['full_validation'] is False
    assert baseline['training']['save_freq'] == 20 and sog['training']['save_freq'] == 10
    assert report['no_training_executed'] and report['sog_status'].startswith('experimental')
    assert json.loads(release.read_text()) == original


@pytest.mark.parametrize('bad', ['partial', 'state', 'model', 'exists', 'wrong-integrity', 'wrong-alignment'])
def test_invalid_inputs_do_not_produce_new_configuration(tmp_path, bad):
    release, data, original, integrity, alignment = inputs(tmp_path)
    status = data / 'conversion_status.json'
    state = json.loads(status.read_text())
    output = tmp_path / 'inputs'
    if bad == 'partial':
        state['status'] = 'partial'
    if bad == 'state':
        state['assumed_charge_spin'] = [0, 2]
    status.write_text(json.dumps(state))
    if bad == 'model':
        original['model']['descriptor']['type'] = 'not-dpa4c'
        release.write_text(json.dumps(original))
    if bad == 'exists':
        output.mkdir()
    if bad == 'wrong-integrity':
        integrity.write_text('{}')
    if bad == 'wrong-alignment':
        alignment.write_text('{}')
    with pytest.raises(ValueError):
        prepare(release, data, output, train_steps=20, sog_steps=10,
                integrity_report=integrity, alignment_report=alignment)
    assert not (output / 'input_dpa4c.json').exists()
