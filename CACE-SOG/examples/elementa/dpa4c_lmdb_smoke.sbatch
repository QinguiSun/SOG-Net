#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=dpa4c-lmdb-smoke

set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune
test -d "$ROOTFS"
test -f "$REPO/CACE-SOG/dpa4c_sog/index_lmdb.py"
echo "NODE=$(hostname)"
echo "REPOSITORY_COMMIT=$(cd "$REPO" && git rev-parse HEAD)"
echo "CONVERTER_SHA256=$(sha256sum "$REPO/CACE-SOG/dpa4c_sog/index_lmdb.py")"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_SMOKE_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune/CACE-SOG
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
PRETRAINED="$ROOT/DPA4C-Neo-OMol25-100M-v20260820.pt"
RUN="$ROOT/runs/dpa4c-lmdb-smoke-$DPA_SMOKE_JOB"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/ase-support-5416043"
export OMP_NUM_THREADS=4
test -r /.singularity.d/libs/libcuda.so.1
echo 'fc42df87b4c81649755e33c64d7e95bc0584a353e01c8f313742b3f73aab412c  '"$PRETRAINED" | sha256sum -c -
mkdir "$RUN"
echo "RUN_DIRECTORY=$RUN"
cd "$REPO"
python -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'
python -u -m dpa4c_sog.index_lmdb --database "$INDEX" --output "$RUN/data" --limit-per-split 16
python - "$ROOT/stratified-pilot-5416082/input_dpa4c_pilot.json" "$RUN" <<'PY'
import json
from pathlib import Path
import sys
config = json.loads(Path(sys.argv[1]).read_text())
run = Path(sys.argv[2])
config['training']['training_data']['systems'] = str(run / 'data/train.lmdb')
config['training']['validation_data']['systems'] = str(run / 'data/val.lmdb')
config['training']['num_steps'] = 10
config['training']['save_freq'] = 10
config['validating'] = {'full_validation': False, 'ema_full_validation': False}
with (run / 'input_lmdb_smoke.json').open('x') as f:
    json.dump(config, f, indent=2)
print('DEBUG_ONLY_LMDB_CONFIG_WRITTEN')
PY
cd "$RUN"
dp --pt-expt train "$RUN/input_lmdb_smoke.json" --finetune "$PRETRAINED"
test -s model.ckpt-10.pt
echo 'LMDB_SMOKE_TRAIN_COMPLETE'
CONTAINER
