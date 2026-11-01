"""Evaluate two DPA4C checkpoints against every train/val pilot frame.

This is a diagnostic on a deliberately biased pilot, not a test-set report.
The DeepMD arrays are in canonical atom order; JSONL forces are restored to
the original extxyz order using the signed pilot manifest's permutation.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import inspect
import json
import math
import os
from pathlib import Path
import sqlite3

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'file changed while hashing: {path}')
    return h.hexdigest()


def checked_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'invalid or missing pilot file: {relative}')
    return path


class Metrics:
    def __init__(self):
        self.n = 0
        self.mean_ref = self.m2_ref = self.sum_error = self.sum_abs = self.sum_sq = 0.0

    def add(self, reference, prediction):
        for r, p in zip(np.asarray(reference).flat, np.asarray(prediction).flat):
            r, p = float(r), float(p)
            if not math.isfinite(r) or not math.isfinite(p):
                raise ValueError('nonfinite prediction or reference')
            self.n += 1
            delta = r - self.mean_ref
            self.mean_ref += delta / self.n
            self.m2_ref += delta * (r - self.mean_ref)
            e = p - r
            self.sum_error += e
            self.sum_abs += abs(e)
            self.sum_sq += e * e

    def result(self):
        if not self.n:
            return None
        return {'count': self.n, 'mean_difference': self.sum_error / self.n,
                'mae': self.sum_abs / self.n, 'rmse': math.sqrt(self.sum_sq / self.n),
                'r2': 1 - self.sum_sq / self.m2_ref if self.n > 1 and self.m2_ref > 0 else None}


def default_loader(path):
    from deepmd.infer import DeepPot
    return DeepPot(str(path), auto_batch_size=False)


def evaluate(pilot, before, after, output, inventory_sha256, frames_sha256,
             before_sha256, after_sha256, loader=default_loader):
    pilot = Path(pilot).resolve(strict=True)
    output = Path(output).resolve()
    before, after = Path(before).resolve(strict=True), Path(after).resolve(strict=True)
    if output.is_relative_to(pilot) or output in (before, after):
        raise ValueError('output must be outside pilot and checkpoint paths')
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / 'status.json'
    def status(data):
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')
        os.replace(temp, status_path)
    status({'status': 'running'})
    try:
        inventory_path = checked_path(pilot, 'pilot_files.json')
        frames_path = checked_path(pilot, 'frames.jsonl')
        status_file = checked_path(pilot, 'pilot_status.json')
        expected = [(inventory_path, inventory_sha256), (frames_path, frames_sha256),
                    (before, before_sha256), (after, after_sha256)]
        for path, digest in expected:
            if sha256(path) != digest.lower():
                raise ValueError(f'SHA256 mismatch: {path}')
        meta = json.loads(status_file.read_text(encoding='utf8'))
        if (meta.get('status') != 'complete' or meta.get('channel') != 'sog_candidates'
                or meta.get('test_split_read') is not False or meta.get('condition') != [0, 1]
                or meta.get('inventory_sha256') != inventory_sha256.lower()
                or meta.get('frames_manifest_sha256') != frames_sha256.lower()):
            raise ValueError('pilot status or provenance does not match requested inputs')
        inventory = json.loads(inventory_path.read_text(encoding='utf8'))
        if inventory.get('format') != 'dpa4c-stratified-pilot-files-v1':
            raise ValueError('unsupported pilot inventory format')
        files = inventory['files']
        for relative, info in files.items():
            path = checked_path(pilot, relative)
            if path.stat().st_size != info['size'] or sha256(path) != info['sha256']:
                raise ValueError(f'pilot file integrity failed: {relative}')
        if files.get('frames.jsonl', {}).get('sha256') != frames_sha256.lower():
            raise ValueError('inventory and frame manifest disagree')
        models = {'before': loader(before), 'after': loader(after)}
        mappings = {}
        for name, model in models.items():
            type_map = list(model.get_type_map())
            if (not type_map or len(set(type_map)) != len(type_map)
                    or not model.has_chg_spin_ebd()
                    or 'charge_spin' not in inspect.signature(model.deep_eval.eval).parameters
                    or int(model.get_dim_fparam()) != 0 or int(model.get_dim_aparam()) != 0):
                raise ValueError(f'{name}: unsupported model type map or charge/spin inputs')
            definition = model.get_model_def_script()
            definition = definition.get('model', definition)
            if definition.get('descriptor', {}).get('type') != 'dpa4c':
                raise ValueError(f'{name}: expected DPA4C descriptor')
            if definition.get('long_range'):
                raise ValueError(f'{name}: expected DPA4C without long-range module')
            mappings[name] = {s: i for i, s in enumerate(type_map)}
        metrics = {split: {name: {'energy': Metrics(), 'force': Metrics()} for name in models}
                   for split in ('train', 'val')}
        top = {split: {name: {'reference': [], 'error': []} for name in models}
               for split in ('train', 'val')}
        counts = {'train': 0, 'val': 0}
        seen = sqlite3.connect(output / 'seen.sqlite3')
        seen.execute('CREATE TABLE frames (split TEXT, system TEXT, row INTEGER, PRIMARY KEY(split, system, row))')
        with frames_path.open(encoding='utf8') as stream, (output / 'predictions.jsonl').open('x', encoding='utf8') as out:
            for line in stream:
                row = json.loads(line)
                split, system, frame = row['split'], row['system'], int(row['row'])
                if split not in counts or row['channel'] != 'sog_candidates' or row['set'] != 'set.000':
                    raise ValueError('unexpected split, channel or set in frame manifest')
                seen.execute('INSERT INTO frames VALUES (?,?,?)', (split, system, frame))
                base = f'{split}/{system}'
                def p(suffix):
                    key = f'{base}/{suffix}'
                    if key not in files:
                        raise ValueError(f'file missing from signed inventory: {key}')
                    return checked_path(pilot, key)
                type_map = p('type_map.raw').read_text(encoding='utf8').split()
                types = [int(v) for v in p('type.raw').read_text(encoding='utf8').split()]
                n = int(row['natoms'])
                if len(types) != n or any(t < 0 or t >= len(type_map) for t in types):
                    raise ValueError('invalid atom types')
                symbols = [type_map[t] for t in types]
                order = row['canonical_to_source']
                if sorted(order) != list(range(n)):
                    raise ValueError('invalid atom-order permutation')
                def array(name):
                    a = np.load(p(f'set.000/{name}.npy'), mmap_mode='r', allow_pickle=False)
                    return np.asarray(a[frame], dtype=np.float64)
                coord, cell = array('coord').reshape(1, n, 3), array('box').reshape(1, 9)
                energy = float(array('energy').reshape(-1)[0])
                ref = array('force').reshape(n, 3)
                if not all(np.isfinite(a).all() for a in (coord, cell, ref)) or not math.isfinite(energy):
                    raise ValueError('nonfinite reference data')
                restored_ref = np.empty_like(ref); restored_ref[order] = ref
                source_symbols = [None] * n
                for i, j in enumerate(order): source_symbols[j] = symbols[i]
                entry = {'identity': row, 'source_symbols': source_symbols, 'reference_energy_eV': energy,
                         'reference_force_eV_A': restored_ref.tolist(), 'models': {}}
                for name, model in models.items():
                    try:
                        mapped = np.array([mappings[name][s] for s in symbols], dtype=np.int32)
                    except KeyError as exc:
                        raise ValueError(f'{name}: missing element {exc}') from exc
                    result = model.eval(coords=coord.copy(), cells=cell.copy(), atom_types=mapped,
                                        atomic=False, charge_spin=np.array([[0., 1.]]))
                    pred_energy = float(np.asarray(result[0]).reshape(-1)[0])
                    pred = np.asarray(result[1], dtype=np.float64).reshape(n, 3)
                    if not math.isfinite(pred_energy) or not np.isfinite(pred).all():
                        raise ValueError(f'{name}: nonfinite model output')
                    restored = np.empty_like(pred); restored[order] = pred
                    metrics[split][name]['energy'].add([energy / n], [pred_energy / n])
                    metrics[split][name]['force'].add(restored_ref, restored)
                    entry['models'][name] = {'energy_eV': pred_energy, 'force_eV_A': restored.tolist(),
                        'max_reference_force_eV_A': float(np.linalg.norm(restored_ref, axis=1).max()),
                        'max_predicted_force_eV_A': float(np.linalg.norm(restored, axis=1).max()),
                        'max_force_error_eV_A': float(np.linalg.norm(restored - restored_ref, axis=1).max())}
                    for kind, key in (('reference', 'max_reference_force_eV_A'),
                                      ('error', 'max_force_error_eV_A')):
                        heap = top[split][name][kind]
                        item = (entry['models'][name][key], counts[split],
                                {'source_id': row['source_id'], 'frame_sha256': row['frame_sha256'],
                                 'value_eV_A': entry['models'][name][key]})
                        if len(heap) < 10: heapq.heappush(heap, item)
                        elif item[:2] > heap[0][:2]: heapq.heapreplace(heap, item)
                out.write(json.dumps(entry, allow_nan=False, separators=(',', ':')) + '\n')
                counts[split] += 1
                if sum(counts.values()) % 50 == 0:
                    print(f'pilot evaluation: {sum(counts.values())} frames', flush=True)
        seen.close()
        if counts != meta['counts']:
            raise ValueError(f'frame counts mismatch: {counts} vs {meta["counts"]}')
        for path, digest in expected:
            if sha256(path) != digest.lower():
                raise ValueError(f'input changed during evaluation: {path}')
        for relative, info in files.items():
            if sha256(checked_path(pilot, relative)) != info['sha256']:
                raise ValueError(f'pilot file changed during evaluation: {relative}')
        summary = {'status': 'complete', 'debug_only': True, 'test_split_read': False,
                   'condition': [0, 1], 'condition_is_not_measured_dft_state': True,
                   'counts': counts, 'inventory_sha256': inventory_sha256,
                   'frames_sha256': frames_sha256, 'before_sha256': before_sha256,
                   'after_sha256': after_sha256, 'mean_difference_definition': 'prediction - reference',
                   'metrics': {split: {name: {'energy_per_atom_eV': values['energy'].result(),
                                             'force_component_eV_A': values['force'].result()}
                                       for name, values in m.items()} for split, m in metrics.items()},
                   'largest_force_frames': {split: {name: {kind: [item[2] for item in sorted(heap, reverse=True)]
                                                      for kind, heap in kinds.items()}
                                                    for name, kinds in names.items()}
                                            for split, names in top.items()},
                   'predictions_sha256': sha256(output / 'predictions.jsonl')}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf8')
        status({'status': 'complete'})
        return summary
    except BaseException as exc:
        status({'status': 'failed', 'error': str(exc), 'results_incomplete': True})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('pilot', 'before', 'after', 'output', 'inventory-sha256', 'frames-sha256',
                 'before-sha256', 'after-sha256'):
        p.add_argument('--' + name, required=True)
    args = p.parse_args()
    result = evaluate(args.pilot, args.before, args.after, args.output, args.inventory_sha256,
                      args.frames_sha256, args.before_sha256, args.after_sha256)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
