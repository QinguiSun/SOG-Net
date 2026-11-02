"""Ordered, bounded multi-frame DeepPot inference, for validation experiments.

Not wired into production. This helper groups only consecutive frames with the
same ordered elements and fully periodic boundary conditions; it holds at most
``batch_size`` pending frames plus one lookahead. The caller supplies ``(identity, atoms)`` pairs and
receives those same objects, plus total energy and forces in source atom order.
It performs no sampling, retries, skipping, writes, or publication of results.
Real SCNet parity tests across multiple compositions are still required.
"""
import numpy as np

from .predict import _check_condition


def ordered_predictions(records, model, details, condition=(0., 1.), *, batch_size=1):
    """Yield ``(identity, atoms, total_energy_eV, forces_eV_A)`` in input order.

    A change in atom order or composition flushes the current batch. Size one
    is the explicit fallback for environments where multi-frame inference has
    not been validated. Backend failures propagate; no frames are discarded.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    cond = np.asarray(condition, dtype=np.float64)
    if cond.shape != (2,) or not np.isfinite(cond).all():
        raise ValueError('condition must contain two finite values')
    type_map = details.get('type_map')
    if not isinstance(type_map, list) or not type_map or any(not isinstance(s, str) or not s for s in type_map) or len(set(type_map)) != len(type_map):
        raise ValueError('invalid model type map')
    mapping = {symbol: i for i, symbol in enumerate(type_map)}
    pending, key = [], None

    def flush():
        size = len(pending)
        if not size:
            return []
        ids, atoms, coordinates, cells, types = zip(*pending)
        n = len(types[0])
        expected_types = np.asarray(types[0], dtype=np.int32)
        if any(tuple(t) != tuple(types[0]) for t in types):
            raise ValueError('mixed ordered atom types in batch')
        output = model.eval(coords=np.stack(coordinates).copy(), cells=np.stack(cells).copy(),
                            atom_types=expected_types.copy(), atomic=False,
                            charge_spin=np.tile(cond, (size, 1)))
        if not isinstance(output, (tuple, list)) or len(output) < 3:
            raise ValueError('DeepPot must return energy, force and virial')
        e, f, v = (np.asarray(x, dtype=np.float64) for x in output[:3])
        if e.shape not in ((size,), (size, 1)) or f.shape != (size, n, 3) or v.shape not in ((size, 9), (size, 3, 3)):
            raise ValueError('unexpected DeepPot energy/force/virial shapes')
        if not all(np.isfinite(x).all() for x in (e, f, v)):
            raise ValueError('nonfinite model output')
        return [(ids[i], atoms[i], float(e.reshape(-1)[i]), f[i].tolist()) for i in range(size)]

    for identity, atoms in records:
        _check_condition(atoms, cond)
        symbols = tuple(atoms.get_chemical_symbols())
        if not symbols or any(s not in mapping for s in symbols):
            raise ValueError('frame has no atoms or an element absent from model type_map')
        types = tuple(mapping[s] for s in symbols)
        if tuple(type_map[i] for i in types) != symbols:
            raise ValueError('atom/type mapping changed atom order')
        pbc = tuple(bool(x) for x in np.asarray(atoms.pbc).reshape(-1))
        if pbc != (True, True, True):
            raise ValueError('fully periodic boundary conditions required')
        coords = np.asarray(atoms.positions, dtype=np.float64)
        cell = np.asarray(atoms.cell, dtype=np.float64)
        if coords.shape != (len(symbols), 3) or cell.shape != (3, 3) or not np.isfinite(coords).all() or not np.isfinite(cell).all():
            raise ValueError('invalid coordinates or cell')
        next_key = (symbols, pbc)
        if pending and (next_key != key or len(pending) == batch_size):
            yield from flush()
            pending.clear()
        key = next_key
        pending.append((identity, atoms, coords.copy(), cell.reshape(9).copy(), types))
    if pending:
        yield from flush()
