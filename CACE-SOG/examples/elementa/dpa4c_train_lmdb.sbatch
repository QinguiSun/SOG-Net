#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dpa4c-lmdb-train

# Usage: sbatch this-file start PREPARED RUN EXPECTED_COMMIT
# Or:    sbatch this-file resume PREPARED RUN EXPECTED_COMMIT CHECKPOINT CHECKPOINT_SHA256
# This script does not establish numerical validity or full-scale memory safety.
set -euo pipefail
MODE=${1:?expected start or resume}
PREPARED=${2:?expected preparation directory}
RUN=${3:?expected run directory}
EXPECTED_COMMIT=${4:?expected full repository commit}
case "$PREPARED" in /work/*) ;; *) echo 'preparation must be under /work'; exit 2;; esac
case "$RUN" in /work/*) ;; *) echo 'run must be under /work'; exit 2;; esac
case "$MODE" in
  start) [ "$#" -eq 4 ] || { echo 'start requires 4 arguments'; exit 2; }; mkdir "$RUN" ;;
  resume) [ "$#" -eq 6 ] || { echo 'resume requires checkpoint and SHA-256'; exit 2; }; test -d "$RUN" ;;
  *) echo 'first argument must be start or resume'; exit 2 ;;
esac
exec 9>"$RUN/.training.lock"
flock -n 9 || { echo 'another training job owns this run'; exit 1; }
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune
test -d "$ROOTFS"
test -x "$ENV_DIR/bin/python"
test -d "$REPO/CACE-SOG/dpa4c_sog"
ACTUAL_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
test "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" || { echo 'repository commit mismatch'; exit 1; }
(cd "$REPO" && git diff --quiet && git diff --cached --quiet) || { echo 'tracked repository changes detected'; exit 1; }
echo "NODE=$(hostname) COMMIT=$ACTUAL_COMMIT RUN=$RUN MODE=$MODE"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU}"
export APPTAINERENV_DPA_TRAIN_MODE="$MODE"
export APPTAINERENV_DPA_TRAIN_PREPARED="$PREPARED"
export APPTAINERENV_DPA_TRAIN_RUN="$RUN"
export APPTAINERENV_DPA_TRAIN_COMMIT="$ACTUAL_COMMIT"
export APPTAINERENV_DPA_TRAIN_JOB="$SLURM_JOB_ID"
if [ "$MODE" = resume ]; then
  export APPTAINERENV_DPA_TRAIN_CHECKPOINT="$5"
  export APPTAINERENV_DPA_TRAIN_CHECKPOINT_SHA="$6"
fi
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune/CACE-SOG
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
PRETRAINED="$ROOT/DPA4C-Neo-OMol25-100M-v20260820.pt"
RELEASE="$ROOT/DPA4C-Neo-OMol25-100M-v20260820.json"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/ase-support-5416043"
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
test -r /.singularity.d/libs/libcuda.so.1
cd "$REPO"
python -c 'import deepmd, torch; print("DeePMD:",deepmd.__version__,"PyTorch:",torch.__version__); assert torch.cuda.is_available(); print("GPU:",torch.cuda.get_device_name(0))'
args=(--mode "$DPA_TRAIN_MODE" --prepared "$DPA_TRAIN_PREPARED" --database "$INDEX"
      --release "$RELEASE" --pretrained "$PRETRAINED" --repo-commit "$DPA_TRAIN_COMMIT" --run "$DPA_TRAIN_RUN")
if [ "$DPA_TRAIN_MODE" = resume ]; then
  args+=(--checkpoint "$DPA_TRAIN_CHECKPOINT" --checkpoint-sha256 "$DPA_TRAIN_CHECKPOINT_SHA")
fi
python -u -m dpa4c_sog.train_run_guard "${args[@]}"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv -l 10 > "$DPA_TRAIN_RUN/gpu-memory-$DPA_TRAIN_JOB.csv" 2>&1 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT
cd "$DPA_TRAIN_RUN"
if [ "$DPA_TRAIN_MODE" = start ]; then
  dp --pt-expt train input_dpa4c_lmdb.json --finetune "$PRETRAINED"
else
  dp --pt-expt train input_dpa4c_lmdb.json --restart "$DPA_TRAIN_CHECKPOINT"
fi
python - <<'PY'
import hashlib, json, re
from pathlib import Path
config = json.loads(Path('input_dpa4c_lmdb.json').read_text())
checkpoints = [(int(m.group(1)), p) for p in Path('.').glob('model.ckpt-*.pt')
               if (m := re.fullmatch(r'model\.ckpt-(\d+)\.pt', p.name))]
if not checkpoints:
    raise SystemExit('No numbered checkpoint produced')
step, path = max(checkpoints, key=lambda x: x[0])
if step < config['training']['num_steps']:
    raise SystemExit('Latest checkpoint step is below requested num_steps')
digest = hashlib.sha256(path.read_bytes()).hexdigest()
result = {'status': 'training_command_complete', 'checkpoint': str(path.resolve()),
          'checkpoint_sha256': digest, 'steps': step, 'best_val_not_selected': True,
          'test_split_read': False}
Path('training_status.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
echo DPA4C_TRAIN_COMMAND_COMPLETE
CONTAINER
