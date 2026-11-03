#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-sampler-b16-coverage
# Diagnostic: sampler indices only, no model training or test-frame reads.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
export APPTAINERENV_COVERAGE_JOB="${SLURM_JOB_ID:?No job ID}"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
RUN="$ROOT/runs/dpa4c-sampler-b16-coverage-$COVERAGE_JOB"
mkdir "$RUN"
python -u - "$ROOT" "$RUN" <<'PY'
import copy, hashlib, json, numbers, resource, sys, time
from pathlib import Path
import numpy as np
import torch, deepmd
from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem
from deepmd.pt_expt.train.training import get_loss, DataRequirementItem
import deepmd.pt_expt.utils.lmdb_dataset as dataset_module
import deepmd.dpmodel.utils.lmdb_data as reader_module

root, run = map(Path, sys.argv[1:])
data = root / 'train-val-lmdb-full-5417356'
train = data / 'train.lmdb'
config_path = root / 'runs/dpa4c-trace-profile-5417519/input.json'
status = json.loads((data / 'status.json').read_text())
if (status.get('status') != 'complete' or status.get('no_sampling') is not True
    or status.get('test_split_read') is not False or status.get('counts', {}).get('train') != 3543786
    or status.get('source_index_metadata_sha256') != 'c07665508e5499fa61dcb5d2faa06ef81fdce40da68f0a88a308ecaf24795251'):
    raise SystemExit('unexpected source status')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
if digest(train / 'data.mdb') != status['train_lmdb_sha256']:
    raise SystemExit('train LMDB SHA mismatch')
if digest(data / 'train.jsonl') != status['train_manifest_sha256']:
    raise SystemExit('train manifest SHA mismatch')
for path, expected in ((Path(dataset_module.__file__), 'ea87d6bd521d23dda2cc047808bc6495a18462f24c13ef70e9cf44761ab738e4'),
                       (Path(reader_module.__file__), '9c7c086335ab5bc1e0dbb8bea6482756e66196468796abd799693f045d4840f0')):
    if digest(path) != expected: raise SystemExit('DeepMD source mismatch: ' + str(path))
cfg = json.loads(config_path.read_text())
model = cfg['model']
if model['descriptor']['type'] != 'dpa4c' or model['descriptor'].get('add_chg_spin_ebd') is not True:
    raise SystemExit('unexpected model descriptor')
if cfg['loss'].get('type', 'ener') != 'ener': raise SystemExit('expected energy loss')
types = model['type_map']
style = cfg['training']['training_data'].get('auto_prob_style')
start = time.monotonic()
system = LmdbDataSystem(str(train), types, batch_size=16, auto_prob_style=style, seed=42,
                        num_workers=0, rank=0, world_size=1)
loss = get_loss(copy.deepcopy(cfg['loss']), 1e-4, len(types), None)
requirements = list(loss.label_requirement)
requirements.append(DataRequirementItem('charge_spin', ndof=2, atomic=False, must=False,
                                        default=np.asarray([0., 1.]), source_policy='default'))
system.add_data_requirements(requirements)
if system._sampler._seed != 42 or system._reader.nframes != 3543786:
    raise SystemExit('unexpected sampler seed or frame count')
init_seconds = time.monotonic() - start
start = time.monotonic()
batch_count = len(system._sampler)
prepare_seconds = time.monotonic() - start
nframes = system._reader.nframes
bits = bytearray((nframes + 7) // 8)
start = time.monotonic()
duplicates = out_of_range = 0
for count in range(batch_count):
    batch = system._batch_iterator._next_indices()
    if not 1 <= len(batch) <= 16 or any(not isinstance(x, numbers.Integral) for x in batch):
        raise ValueError('invalid batch')
    for value in batch:
        index = int(value)
        if not 0 <= index < nframes:
            out_of_range += 1
            continue
        mask = 1 << (index & 7); pos = index >> 3
        if bits[pos] & mask: duplicates += 1
        bits[pos] |= mask
    if (count + 1) % 250000 == 0:
        print(f'indices: {count+1:,}/{batch_count:,}; seconds={time.monotonic()-start:.1f}', flush=True)
walk_seconds = time.monotonic() - start
unique = sum(b.bit_count() for b in bits)
report = {'status': 'passed' if unique == nframes and not duplicates and not out_of_range else 'failed',
          'scope': 'single-rank sampler epoch indices only; no frame decoding or model updates',
          'no_training_executed': True, 'test_split_read': False, 'seed': 42, 'batch_size': 16,
          'auto_prob_style': style, 'frames': nframes, 'batches': batch_count, 'unique_indices': unique,
          'duplicates': duplicates, 'out_of_range': out_of_range, 'missing': nframes - unique,
          'initialization_seconds': init_seconds, 'sampler_materialization_seconds': prepare_seconds,
          'iteration_seconds': walk_seconds, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'config_sha256': digest(config_path), 'train_lmdb_sha256': status['train_lmdb_sha256'],
          'numpy': np.__version__, 'torch': torch.__version__, 'deepmd': deepmd.__version__}
(run / 'coverage.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
system.close()
if report['status'] != 'passed': raise SystemExit('sampler coverage failed')
print('FULL_TRAIN_B16_SAMPLER_COVERAGE_OK', flush=True)
PY
CONTAINER
