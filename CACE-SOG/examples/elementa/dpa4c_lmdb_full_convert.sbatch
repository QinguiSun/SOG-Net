#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dpa4c-lmdb-full-convert

set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune
CONVERTER="$REPO/CACE-SOG/dpa4c_sog/index_lmdb_resume.py"
test -d "$ROOTFS"
test -f "$CONVERTER"
echo '76de3029742ce856eabb50e533cdc9ac42e817ddb8692a931e9d2c4e5919f914  '"$CONVERTER" | sha256sum -c -
echo "NODE=$(hostname)"
echo "REPOSITORY_COMMIT=$(cd "$REPO" && git rev-parse HEAD)"
df -h "$ROOT"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_FULL_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune/CACE-SOG
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
OUTPUT="$ROOT/train-val-lmdb-full-$DPA_FULL_JOB"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/ase-support-5416043:$REPO"
export OMP_NUM_THREADS=2
if [ -e "$OUTPUT" ]; then echo "Output already exists; refusing to reuse it: $OUTPUT"; exit 1; fi
echo "FULL_LMDB_DIRECTORY=$OUTPUT"
cd "$REPO"
python -u -m dpa4c_sog.index_lmdb_resume --database "$INDEX" --output "$OUTPUT" --commit-every 1000
echo "FULL_LMDB_CONVERSION_COMPLETE"
CONTAINER
