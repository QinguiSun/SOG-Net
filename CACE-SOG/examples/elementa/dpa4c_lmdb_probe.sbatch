#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=dpa4c-lmdb-probe

set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
test -d "$ROOTFS"
test -x "$ENV_DIR/bin/python"
mkdir -p "$ROOT/reports"
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_PROBE_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
ASE_DIR="$ROOT/ase-support-5416043"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
test -d "$ASE_DIR/ase"
export PYTHONPATH="$ASE_DIR"
python - "$ROOT/reports/dpa4c-lmdb-probe-$DPA_PROBE_JOB.json" <<'PY'
import importlib
import inspect
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
if out.exists():
    raise SystemExit('Probe report already exists: ' + str(out))
report = {'status': 'inspection_only', 'python': sys.version, 'modules': {}}
for name in ('torch', 'deepmd', 'numpy', 'ase', 'lmdb', 'msgpack', 'dpdata',
             'deepmd.pt_expt.utils.lmdb_dataset',
             'deepmd.pt_expt.entrypoints.main',
             'deepmd.dpmodel.utils.lmdb_data',
             'dpdata.formats.deepmd.lmdb.format'):
    try:
        obj = importlib.import_module(name)
        item = {'imported': True, 'version': getattr(obj, '__version__', None),
                'path': getattr(obj, '__file__', None)}
        if name == 'torch':
            item['cuda_version'] = obj.version.cuda
            item['cuda_available'] = obj.cuda.is_available()
        for symbol in ('LmdbDataSystem', 'LmdbDataReader', 'dump_systems', '_build_data_system'):
            if hasattr(obj, symbol):
                value = getattr(obj, symbol)
                try:
                    signature = str(inspect.signature(value))
                except (TypeError, ValueError):
                    signature = 'unavailable'
                item[symbol] = signature
        if hasattr(obj, '_build_data_system'):
            try:
                item['builder_mentions_lmdb'] = 'lmdb' in inspect.getsource(obj._build_data_system).lower()
            except (TypeError, OSError):
                item['builder_mentions_lmdb'] = None
        report['modules'][name] = item
    except Exception as exc:
        report['modules'][name] = {'imported': False, 'error': str(exc)}
report['scope'] = 'Imports and signatures only; no dataset conversion, training or evaluation.'
with out.open('x', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False)
    stream.write('\n')
print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
print('LMDB_INSPECTION_COMPLETE', flush=True)
PY
CONTAINER
