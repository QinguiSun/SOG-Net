#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=dpa4c-restricted-start
# Experimental 20-step debug-only test; does not read test or train full data.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
CODE="$ROOT/replay-experiment-20261102/restricted-code"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?GPU missing}"
export APPTAINERENV_RESTRICTED_JOB="${SLURM_JOB_ID:?Job ID missing}"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOT/ubuntu-jammy-rootfs-5414994" /bin/bash -s <<'SCRIPT'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
CODE="$ROOT/replay-experiment-20261102/restricted-code"
CFG="$ROOT/runs/dpa4c-epoch-fault-probe-5417774/input.json"
DATA="$ROOT/runs/dpa4c-lmdb-recover-5416703/data"
INITIAL="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-restricted-start-$RESTRICTED_JOB"
mkdir "$RUN"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE"
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
python -u -m dpa4c_sog.restricted_train_experimental --mode start --config "$CFG" --run "$RUN" --train "$DATA/train.lmdb" --val "$DATA/val.lmdb" --initial "$INITIAL" --initial-sha256 d6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355
test -s "$RUN/restricted_complete.json"
echo "RUN=$RUN"
echo RESTRICTED_DEBUG_START_COMPLETE
SCRIPT
