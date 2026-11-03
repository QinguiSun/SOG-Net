#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=dpa4c-eval-batch
# Experimental: first 8 val frames only; no test reads and no training.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?no allocated GPU}"
export APPTAINERENV_PROBE_JOB="${SLURM_JOB_ID:?no job ID}"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOT/ubuntu-jammy-rootfs-5414994" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO="$ROOT/replay-experiment-20261102/val-code"
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
MODEL="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-eval-batch-$PROBE_JOB"
echo 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355  '"$MODEL" | sha256sum -c -
mkdir "$RUN"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/ase-support-5416043:$REPO"
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
cd "$REPO"
python -u - "$INDEX" "$MODEL" "$RUN" <<'PY'
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from cace_sog_training.data import IndexReader
from dpa4c_sog.predict import _default_loader, _model_details, _check_condition

database, checkpoint, run = map(Path, sys.argv[1:])
if not torch.cuda.is_available(): raise RuntimeError('GPU unavailable')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
model = _default_loader(checkpoint)
details = _model_details(model, 'dpa')
conditions = np.array([0., 1.], dtype=np.float64)
frames, ids = [], []
with IndexReader(database) as reader:
    for ordinal in range(8):
        rec = reader.record('sog_candidates', 'val', ordinal)
        atom = reader.read_atoms(rec)
        _check_condition(atom, conditions)
        if not atom.pbc.all() or not np.isfinite(atom.positions).all() or not np.isfinite(atom.cell.array).all():
            raise ValueError('nonperiodic or nonfinite frame')
        frames.append(atom)
        ids.append([ordinal, rec['id'], rec['sha256']])
    reader.verify_sources()
symbols = frames[0].get_chemical_symbols()
if not symbols or any(a.get_chemical_symbols() != symbols for a in frames):
    raise ValueError('probe requires identical ordered atom types in all 8 frames')
types = np.asarray([details['type_map'].index(s) for s in symbols], dtype=np.int32)
n = len(types)
coords = np.stack([a.positions for a in frames]).astype(np.float64)
cells = np.stack([a.cell.array.reshape(9) for a in frames]).astype(np.float64)
def evaluate(c, box):
    size = len(c)
    out = model.eval(coords=c.copy(), cells=box.copy(), atom_types=types.copy(),
                     atomic=False, charge_spin=np.tile(conditions, (size, 1)))
    energy = np.asarray(out[0], dtype=np.float64)
    force = np.asarray(out[1], dtype=np.float64)
    virial = np.asarray(out[2], dtype=np.float64)
    if energy.shape not in ((size,), (size, 1)) or force.shape != (size, n, 3) or virial.shape not in ((size, 9), (size, 3, 3)):
        raise ValueError(f'unexpected output shapes: {energy.shape}, {force.shape}, {virial.shape}')
    if not all(np.isfinite(x).all() for x in (energy, force, virial)):
        raise ValueError('nonfinite output')
    return energy.reshape(-1), force
def timed(size):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t = time.monotonic(); energies, forces = [], []
    for j in range(0, 8, size):
        e, f = evaluate(coords[j:j+size], cells[j:j+size])
        energies.extend(e.tolist()); forces.extend(f.tolist())
    torch.cuda.synchronize()
    return (np.asarray(energies), np.asarray(forces), time.monotonic()-t,
            torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
# Prime the identical B=1 reference first. Timing excludes this first warmup.
evaluate(coords[:1], cells[:1]); torch.cuda.synchronize()
base_e, base_f, base_t, base_mem, base_reserved = timed(1)
result = {'debug_only': True, 'test_split_read': False, 'frames': ids,
          'symbols': symbols, 'condition': conditions.tolist(), 'model': str(checkpoint),
          'single': {'seconds': base_t, 'peak_allocated_bytes': base_mem,
                     'peak_reserved_bytes': base_reserved}, 'batches': {}}
for batch_size in (2, 4, 8):
    # Include the first invocation for the shape in timing; no unmeasured compilation.
    e, f, elapsed, memory, reserved = timed(batch_size)
    de = float(np.max(np.abs(e-base_e)) / n)
    df = float(np.max(np.abs(f-base_f)))
    warm_e, warm_f, warm_seconds, warm_memory, warm_reserved = timed(batch_size)
    if np.max(np.abs(warm_e-base_e))/n > 1e-3 or np.max(np.abs(warm_f-base_f)) > 1e-3:
        raise ValueError('warm batch/single differences exceed diagnostic tolerance')
    result['batches'][str(batch_size)] = {'first_seconds': elapsed, 'warm_seconds': warm_seconds,
        'peak_allocated_bytes': max(memory, warm_memory),
        'peak_reserved_bytes': max(reserved, warm_reserved),
        'max_energy_difference_eV_per_atom': de, 'max_force_difference_eV_A': df}
    print('BATCH_RESULT', batch_size, result['batches'][str(batch_size)], flush=True)
    if de > 1e-3 or df > 1e-3:
        raise ValueError('batch/single differences exceed diagnostic tolerance')
# Check one atomic periodic image in the batched path, respecting each frame cell.
moved = coords.copy(); moved[0, 0] += cells[0].reshape(3, 3)[0]
shift_e, shift_f = evaluate(moved, cells)
torch.cuda.synchronize()
pe = float(np.max(np.abs(shift_e-base_e)))
pf = float(np.max(np.abs(shift_f-base_f)))
result['periodic_image'] = {'max_energy_difference_eV': pe, 'max_force_difference_eV_A': pf}
if pe > 2e-3 or pf > 2e-3: raise ValueError('batch periodic image check failed')
result['status'] = 'passed'
(run/'batch_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(result, indent=2), flush=True)
print('DPA4C_BATCH_VAL_DIAGNOSTIC_PASSED', flush=True)
PY
CONTAINER
