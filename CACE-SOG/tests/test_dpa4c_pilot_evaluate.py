import json

import numpy as np
import pytest

from dpa4c_sog.pilot_evaluate import evaluate, sha256
from dpa4c_sog.stratified_pilot import prepare_pilot
from test_dpa4c_convert import make_dataset
from test_dpa4c_index_smoke import release_file


class FakeModel:
    def __init__(self, shift):
        self.shift = shift
        self.deep_eval = self

    def has_chg_spin_ebd(self): return True
    def get_dim_fparam(self): return 0
    def get_dim_aparam(self): return 0
    def get_model_def_script(self): return {'descriptor': {'type': 'dpa4c'}}

    def get_type_map(self):
        from ase.data import chemical_symbols
        return chemical_symbols[1:]

    def eval(self, coords, cells, atom_types, atomic, charge_spin):
        assert atomic is False
        assert charge_spin.tolist() == [[0.0, 1.0]]
        n = coords.shape[1]
        return np.array([[self.shift * n]]), np.full((1, n, 3), self.shift), np.zeros((1, 9))


def inputs(tmp_path):
    _, db = make_dataset(tmp_path)
    config = tmp_path / 'release.json'
    release_file(config)
    pilot = tmp_path / 'pilot'
    prepare_pilot(db, config, pilot, per_element=2, seed=7, steps=9)
    before = tmp_path / 'before.pt'; before.write_bytes(b'before')
    after = tmp_path / 'after.pt'; after.write_bytes(b'after')
    return (pilot, before, after, sha256(pilot / 'pilot_files.json'),
            sha256(pilot / 'frames.jsonl'), sha256(before), sha256(after))


def run(args, output):
    pilot, before, after, inventory, frames, a, b = args
    return evaluate(pilot, before, after, output, inventory, frames, a, b,
                    loader=lambda path: FakeModel(1 if path == before else 2))


def test_complete_paired_metrics_and_original_order(tmp_path):
    args = inputs(tmp_path)
    output = tmp_path / 'evaluation'
    summary = run(args, output)
    assert summary['status'] == 'complete'
    assert summary['test_split_read'] is False
    records = [json.loads(line) for line in (output / 'predictions.jsonl').read_text().splitlines()]
    assert len(records) == sum(summary['counts'].values())
    for split in ('train', 'val'):
        selected = [r for r in records if r['identity']['split'] == split]
        if not selected:
            continue
        ref = np.array([r['reference_energy_eV'] / r['identity']['natoms'] for r in selected])
        metric = summary['metrics'][split]['before']['energy_per_atom_eV']
        assert metric['count'] == len(selected)
        assert metric['mean_difference'] == pytest.approx(np.mean(1 - ref))
        assert metric['mae'] == pytest.approx(np.mean(np.abs(1 - ref)))
        assert metric['rmse'] == pytest.approx(np.sqrt(np.mean((1 - ref) ** 2)))
        for r in selected:
            original_order = r['identity']['canonical_to_source']
            base = args[0] / split / r['identity']['system'] / 'set.000'
            canonical = np.load(base / 'force.npy')[r['identity']['row']].reshape(-1, 3)
            restored = np.empty_like(canonical); restored[original_order] = canonical
            np.testing.assert_allclose(r['reference_force_eV_A'], restored)
            assert r['models']['after']['max_force_error_eV_A'] == pytest.approx(
                np.linalg.norm(2 - restored, axis=1).max())


def test_tampered_input_fails_and_marks_incomplete(tmp_path):
    args = inputs(tmp_path)
    (args[0] / 'frames.jsonl').write_text('tampered')
    output = tmp_path / 'bad'
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        run(args, output)
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
    assert not (output / 'summary.json').exists()


def test_existing_output_is_never_overwritten(tmp_path):
    args = inputs(tmp_path)
    output = tmp_path / 'exists'; output.mkdir()
    with pytest.raises(FileExistsError):
        run(args, output)
