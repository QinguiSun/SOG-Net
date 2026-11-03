#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-restart-probe
# Diagnostic only. No test frames. The source LMDBs are opened read-only.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
test -d "$ROOTFS"
echo "NODE=$(hostname)"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU}"
export APPTAINERENV_DPA_RESTART_PROBE_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
DATA="$ROOT/train-val-lmdb-full-5417356"
CALIBRATED="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
SOURCE_CONFIG="$ROOT/runs/dpa4c-trace-profile-5417519/input.json"
RUN="$ROOT/runs/dpa4c-restart-probe-$DPA_RESTART_PROBE_JOB"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
test -r /.singularity.d/libs/libcuda.so.1
echo 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355  '"$CALIBRATED" | sha256sum -c -
mkdir "$RUN"
echo "RESTART_PROBE_DIRECTORY=$RUN"
python - "$SOURCE_CONFIG" "$DATA" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
source, data, run = map(Path, sys.argv[1:])
status = json.loads((data / 'status.json').read_text())
assert status['status'] == 'complete' and status['no_sampling'] is True and status['test_split_read'] is False
assert status['counts'] == {'train': 3543786, 'val': 443734}
assert status['source_index_metadata_sha256'] == 'c07665508e5499fa61dcb5d2faa06ef81fdce40da68f0a88a308ecaf24795251'
for split in ('train', 'val'):
    digest = hashlib.sha256()
    with (data / (split + '.lmdb') / 'data.mdb').open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): digest.update(block)
    if digest.hexdigest() != status[split + '_lmdb_sha256']:
        raise ValueError(split + ' LMDB SHA mismatch')
cfg = json.loads(source.read_text())
assert cfg['model']['descriptor']['type'] == 'dpa4c' and cfg['model']['descriptor']['rcut'] == 6.0
for split, field in (('train', 'training_data'), ('val', 'validation_data')):
    assert Path(cfg['training'][field]['systems']).resolve() == (data / (split + '.lmdb')).resolve()
    cfg['training'][field]['batch_size'] = 1
cfg['training'].update(num_steps=6, save_freq=3, max_ckpt_keep=10, disp_freq=1,
                       enable_compile=False, enable_ema=False)
cfg['training']['validation_data']['numb_batch'] = 1
cfg['validating'] = {'full_validation': False, 'ema_full_validation': False}
with (run / 'input.json').open('x') as f: json.dump(cfg, f, indent=2)
(run / 'continuous').mkdir(); (run / 'resumed').mkdir()
print('RESTART_PROBE_SKIPS_NEIGHBOR_STAT_ONLY_FOR_DIAGNOSIS', flush=True)
PY
cat > "$RUN/instrument.py" <<'PY'
import json, os, runpy, sys
from pathlib import Path
import numpy as np
from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem

original = LmdbDataSystem.get_batch
log = Path('batch-fids.jsonl')
def instrument(self, *args, **kwargs):
    batch = original(self, *args, **kwargs)
    candidates = [getattr(self, 'lmdb_path', None)]
    for name in ('reader', '_reader', 'dataset', '_dataset'):
        obj = getattr(self, name, None)
        candidates += [getattr(obj, 'lmdb_path', None)]
    paths = [str(p) for p in candidates if p]
    split = next((s for s in ('train', 'val') if any(p.endswith('/' + s + '.lmdb') for p in paths)), None)
    fids = batch.get('fid')
    if split is None or fids is None:
        raise RuntimeError('cannot identify LMDB split and frame IDs; refusing an unverifiable result')
    values = [int(x) for x in np.asarray(fids).reshape(-1)]
    with log.open('a', encoding='utf8') as f:
        f.write(json.dumps({'split': split, 'fids': values}) + '\n'); f.flush(); os.fsync(f.fileno())
    return batch
LmdbDataSystem.get_batch = instrument
dp, config, operation, checkpoint = sys.argv[1:]
sys.argv = [dp, '--pt-expt', 'train', config, operation, checkpoint, '--skip-neighbor-stat']
runpy.run_path(dp, run_name='__main__')
PY
cd "$RUN/continuous"
python -u "$RUN/instrument.py" "$ENV_DIR/bin/dp" "$RUN/input.json" --init-model "$CALIBRATED"
test -s model.ckpt-3.pt
test -s model.ckpt-6.pt
cd "$RUN/resumed"
python -u "$RUN/instrument.py" "$ENV_DIR/bin/dp" "$RUN/input.json" --restart "$RUN/continuous/model.ckpt-3.pt"
test -s model.ckpt-6.pt
python - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
def train_fids(path):
    return [r['fids'] for line in path.read_text().splitlines()
            if (r := json.loads(line))['split'] == 'train']
continuous = train_fids(root / 'continuous/batch-fids.jsonl')
resumed = train_fids(root / 'resumed/batch-fids.jsonl')
if len(continuous) != 6 or len(resumed) != 3:
    raise ValueError(f'unexpected batch record counts: continuous={len(continuous)} resumed={len(resumed)}')
def state(path):
    o = torch.load(path, map_location='cpu', weights_only=True)
    return o.get('model', o)
left, right = state(root / 'continuous/model.ckpt-6.pt'), state(root / 'resumed/model.ckpt-6.pt')
keys = [k for k in left.keys() & right.keys() if isinstance(left[k], torch.Tensor) and isinstance(right[k], torch.Tensor) and left[k].shape == right[k].shape]
if not keys: raise ValueError('no comparable tensors')
changed = [k for k in keys if not torch.equal(left[k], right[k])]
result = {'status': 'complete', 'debug_only': True, 'test_split_read': False,
          'continuous_fids': continuous, 'expected_resume_fids': continuous[3:], 'observed_resume_fids': resumed,
          'same_next_fids': continuous[3:] == resumed, 'matching_tensor_keys': len(keys), 'different_tensor_keys': len(changed),
          'notes': 'Exact tensor equality is reported descriptively; nondeterminism can also cause differences. FIDs are decisive for data-order comparison.'}
with (root / 'comparison.json').open('x') as f: json.dump(result, f, indent=2)
print(json.dumps(result, indent=2), flush=True)
PY
echo 'DPA4C_RESTART_DATA_ORDER_PROBE_COMPLETE'
CONTAINER
