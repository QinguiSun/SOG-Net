import numpy as np
import pytest
from ase import Atoms

from dpa4c_sog.batched_infer_experimental import ordered_predictions


class Model:
    def __init__(self, invalid=None):
        self.calls = []
        self.invalid = invalid

    def eval(self, coords, cells, atom_types, atomic, charge_spin):
        size, n, _ = coords.shape
        assert cells.shape == (size, 9)
        assert not atomic
        np.testing.assert_array_equal(charge_spin, np.tile([0, 1], (size, 1)))
        self.calls.append((size, atom_types.tolist()))
        energy = coords.sum(axis=(1, 2))[:, None] + atom_types.sum()
        force = coords + atom_types[None, :, None]
        virial = np.zeros((size, 9))
        if self.invalid == 'shape': force = force.reshape(size, -1)
        if self.invalid == 'nan': force[0, 0, 0] = float('nan')
        return energy, force, virial


def atom(symbols, x):
    a = Atoms(symbols, positions=np.arange(len(symbols) * 3).reshape(-1, 3) + x,
              pbc=True, cell=np.eye(3) * 10)
    a.arrays['REF_forces'] = np.ones((len(symbols), 3)) * x
    return a


def test_groups_consecutive_identical_order_and_preserves_each_record():
    records = [(f'id-{i}', atom(s, i)) for i, s in enumerate([
        ['Na', 'Cl'], ['Na', 'Cl'], ['Na', 'Cl'], ['Cl', 'Na'],
        ['Na', 'Cl'], ['Na'], ['Na']])]
    details = {'type_map': ['Cl', 'Na']}
    batched, single = Model(), Model()
    out = list(ordered_predictions(iter(records), batched, details, batch_size=2))
    ref = list(ordered_predictions(iter(records), single, details, batch_size=1))
    assert [x[0] for x in out] == [x[0] for x in records]
    assert [x[1] for x in out] == [x[1] for x in records]
    assert [call[0] for call in batched.calls] == [2, 1, 1, 1, 2]
    assert batched.calls[0][1] == [1, 0] and batched.calls[2][1] == [0, 1]
    for a, b in zip(out, ref):
        assert a[2] == b[2]
        np.testing.assert_allclose(a[3], b[3])
        np.testing.assert_array_equal(a[1].arrays['REF_forces'], b[1].arrays['REF_forces'])


@pytest.mark.parametrize('bad', ['shape', 'nan'])
def test_rejects_invalid_model_output(bad):
    with pytest.raises(ValueError):
        list(ordered_predictions([(1, atom(['Na'], 0)), (2, atom(['Na'], 1))],
             Model(bad), {'type_map': ['Na']}, batch_size=2))


def test_rejects_bad_inputs_and_condition():
    m, d = Model(), {'type_map': ['Na']}
    with pytest.raises(ValueError, match='positive integer'):
        list(ordered_predictions([], m, d, batch_size=0))
    with pytest.raises(ValueError, match='type map'):
        list(ordered_predictions([], m, {'type_map': ['Na', 'Na']}))
    with pytest.raises(ValueError, match='absent'):
        list(ordered_predictions([(1, atom(['Cl'], 0))], m, d))
    a = atom(['Na'], 0); a.pbc = (True, False, True)
    with pytest.raises(ValueError, match='periodic'):
        list(ordered_predictions([(1, a)], m, d))
    a = atom(['Na'], 0); a.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match='coordinates'):
        list(ordered_predictions([(1, a)], m, d))
    with pytest.raises(ValueError, match='finite'):
        list(ordered_predictions([], m, d, condition=(0, np.nan)))


def test_empty_and_backend_errors_never_skip_frames():
    m, d = Model(), {'type_map': ['Na']}
    assert list(ordered_predictions([], m, d, batch_size=2)) == []
    assert not m.calls
    class Broken(Model):
        def eval(self, **kwargs): raise RuntimeError('backend failure')
    with pytest.raises(RuntimeError, match='backend failure'):
        list(ordered_predictions([(0, atom(['Na'], 0))], Broken(), d))
