#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=dpa4c-full-100

# 100-step full-LMDB startup and checkpoint diagnostic. The fixed DPA4C descriptor is
# checked and neighbor-stat is skipped only in this diagnostic. Tracebacks
# identify Python stack locations, not CUDA kernel time; disp_freq=10 adds periodic validation.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
test -d "$ROOTFS"
echo "PROFILE_NODE=$(hostname)"
echo "PROFILE_HOST_START=$(date -Is)"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_PROFILE_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
FULL="$ROOT/train-val-lmdb-full-5417356"
SOURCE="$ROOT/runs/dpa4c-lmdb-scale-5416702"
CALIBRATED="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-full-100-$DPA_PROFILE_JOB"
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
echo "PROFILE_RUN_DIRECTORY=$RUN"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 10 > "$RUN/gpu-monitor.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
python - "$SOURCE" "$FULL" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
source, full, run = map(Path, sys.argv[1:])
status_path = full / 'status.json'
status = json.loads(status_path.read_text())
if not (status.get('status') == 'complete' and status.get('debug_only') is False
        and status.get('no_sampling') is True and status.get('test_split_read') is False
        and status.get('counts') == {'train': 3543786, 'val': 443734}
        and status.get('source_index_metadata_sha256') == 'c07665508e5499fa61dcb5d2faa06ef81fdce40da68f0a88a308ecaf24795251'
        and status.get('condition') == [0, 1]):
    raise SystemExit('Expected complete unsampled full train/val LMDB data')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
print('PROFILE_FULL_STATUS_SHA256=' + sha(status_path), flush=True)
for split in ('train', 'val'):
    path = full / (split + '.lmdb') / 'data.mdb'
    if sha(path) != status[split + '_lmdb_sha256']:
        raise SystemExit('Dataset hash mismatch: ' + str(path))
    print('PROFILE_' + split.upper() + '_LMDB_SHA256=' + status[split + '_lmdb_sha256'], flush=True)
config_path = source / 'input_lmdb_smoke.json'
config = json.loads(config_path.read_text())
descriptor = config.get('model', {}).get('descriptor', {})
if descriptor.get('type') != 'dpa4c' or descriptor.get('rcut') != 6.0:
    raise SystemExit('Neighbor-stat skip requires the fixed DPA4C 6.0 Å descriptor')
for split in ('train', 'val'):
    field = 'training_data' if split == 'train' else 'validation_data'
    config['training'][field]['systems'] = str((full / (split + '.lmdb')).resolve(strict=True))
config['training']['training_data']['batch_size'] = 1
config['training']['validation_data']['batch_size'] = 1
config['training']['validation_data']['numb_batch'] = 1
config['training']['num_steps'] = 100
config['training']['save_freq'] = 50
config['training']['disp_freq'] = 10
config['training']['enable_compile'] = False
config['validating'] = {'full_validation': False, 'ema_full_validation': False}
out = run / 'input.json'
with out.open('x') as f:
    json.dump(config, f, indent=2)
print('PROFILE_SOURCE_CONFIG_SHA256=' + sha(config_path), flush=True)
print('PROFILE_CONFIG_SHA256=' + sha(out), flush=True)
print('PROFILE_SKIPS_NEIGHBOR_STAT_FIXED_DPA4C_RCUT_6_A', flush=True)
PY
cd "$RUN"
echo "PROFILE_START=$(date -Is)"
python -u - "$ENV_DIR/bin/dp" "$RUN/input.json" "$CALIBRATED" <<'PY'
import faulthandler
import resource
import runpy
import sys
import threading
from datetime import datetime
dp, config, model = sys.argv[1:]
print('PROFILE_WRAPPER_ENTER=' + datetime.now().astimezone().isoformat(), flush=True)
faulthandler.enable(all_threads=True)
faulthandler.dump_traceback_later(30, repeat=True)
stop = threading.Event()
def sample_rss():
    with open('process-rss.csv', 'x', encoding='utf-8') as out:
        out.write('timestamp,vmrss_kib\n')
        while not stop.is_set():
            try:
                with open('/proc/self/status', encoding='utf-8') as status:
                    rss = next((line.split()[1] for line in status if line.startswith('VmRSS:')), '')
                out.write(datetime.now().astimezone().isoformat() + ',' + rss + '\n')
                out.flush()
            except OSError:
                pass
            stop.wait(10)
thread = threading.Thread(target=sample_rss, daemon=True)
thread.start()
sys.argv = [dp, '--pt-expt', 'train', config, '--init-model', model, '--skip-neighbor-stat']
try:
    runpy.run_path(dp, run_name='__main__')
finally:
    faulthandler.cancel_dump_traceback_later()
    stop.set()
    thread.join(timeout=2)
    print('PROFILE_RU_MAXRSS_KIB=' + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), flush=True)
    print('PROFILE_WRAPPER_EXIT=' + datetime.now().astimezone().isoformat(), flush=True)
PY
test -s model.ckpt-100.pt
echo "PROFILE_FINISH=$(date -Is)"
echo 'DPA4C_FULL_100_DIAGNOSTIC_COMPLETE'
CONTAINER
