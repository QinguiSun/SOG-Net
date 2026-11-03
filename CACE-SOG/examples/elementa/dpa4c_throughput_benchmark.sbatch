#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=dpa4c-throughput

# Diagnostic benchmark only: 12 optimizer steps per batch size on the same
# previously prepared 10k-frame train/val datasets. The full test set is never
# opened. Job timings include training/periodic-validation and checkpoint I/O.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
test -d "$ROOTFS"
echo "NODE=$(hostname)"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_BENCH_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
ORIGINAL="$ROOT/runs/dpa4c-lmdb-scale-5416702"
CALIBRATED="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-throughput-$DPA_BENCH_JOB"
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
echo "RUN_DIRECTORY=$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 > "$RUN/gpu-monitor.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python - "$ORIGINAL" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
source, run = map(Path, sys.argv[1:])
with (source / 'data/status.json').open() as f:
    status = json.load(f)
if not (status.get('status') == 'complete' and status.get('debug_only') is True
        and status.get('test_split_read') is False and status.get('counts') == {'train': 10000, 'val': 10000}
        and status.get('condition') == [0, 1]):
    raise SystemExit('Expected the complete 10k/split debugging dataset')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
for split in ('train', 'val'):
    path = source / 'data' / (split + '.lmdb') / 'data.mdb'
    if sha(path) != status[split + '_lmdb_sha256']:
        raise SystemExit('Dataset hash mismatch: ' + str(path))
config_path = source / 'input_lmdb_smoke.json'
config = json.loads(config_path.read_text())
for split in ('train', 'val'):
    field = 'training_data' if split == 'train' else 'validation_data'
    actual = Path(config['training'][field]['systems']).resolve()
    if actual != (source / 'data' / (split + '.lmdb')).resolve():
        raise SystemExit('Unexpected source dataset: ' + str(actual))
print('BENCHMARK_SOURCE_CONFIG_SHA256=' + sha(config_path), flush=True)
for batch in (1, 8):
    d = run / ('batch-' + str(batch))
    d.mkdir()
    # Reload so the second config cannot inherit an accidental mutation.
    cfg = json.loads(config_path.read_text())
    cfg['training']['training_data']['batch_size'] = batch
    cfg['training']['validation_data']['batch_size'] = batch
    cfg['training']['validation_data']['numb_batch'] = 1
    cfg['training']['num_steps'] = 12
    cfg['training']['save_freq'] = 12
    cfg['training']['disp_freq'] = 5
    cfg['training']['enable_compile'] = False
    cfg['validating'] = {'full_validation': False, 'ema_full_validation': False}
    out = d / 'input.json'
    with out.open('x') as f:
        json.dump(cfg, f, indent=2)
    print('CONFIG batch=' + str(batch) + ' sha256=' + sha(out), flush=True)
PY
for batch in 1 8; do
  cd "$RUN/batch-$batch"
  echo "BATCH_START batch=$batch unix=$(date +%s)"
  dp --pt-expt train input.json --init-model "$CALIBRATED"
  test -s model.ckpt-12.pt
  echo "BATCH_FINISH batch=$batch unix=$(date +%s)"
done
echo DPA4C_THROUGHPUT_BENCHMARK_COMPLETE
CONTAINER
