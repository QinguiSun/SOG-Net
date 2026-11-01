#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-pilot-eval

set -euo pipefail
DPA_ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$DPA_ROOT/ubuntu-jammy-rootfs-5414994"
ENV_DIR="$DPA_ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net
test -d "$ROOTFS"
test -x "$ENV_DIR/bin/python"
test -d "$REPO/CACE-SOG/dpa4c_sog"
echo "Compute node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:?Slurm did not provide a GPU assignment}"
echo "Repository revision: $(cd "$REPO" && git rev-parse HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_DPA_EVAL_JOB="$SLURM_JOB_ID"

apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
DPA_ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
DPA_ENV="$DPA_ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
set +u
source "$DPA_ENV/bin/activate" "$DPA_ENV"
set -u
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
test -r /.singularity.d/libs/libcuda.so.1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
cd /work/home/einlovestan/projects/SOG-Net/CACE-SOG
python -c 'import torch; print("CUDA available:", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'
OUTPUT="$DPA_ROOT/runs/dpa4c-pilot-eval-$DPA_EVAL_JOB"
echo "EVALUATION_DIRECTORY=$OUTPUT"
python -u -m dpa4c_sog.pilot_evaluate \
  --pilot "$DPA_ROOT/stratified-pilot-5416082" \
  --before "$DPA_ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt" \
  --after "$DPA_ROOT/runs/dpa4c-pilot-train-5416167/model.ckpt-200.pt" \
  --output "$OUTPUT" \
  --inventory-sha256 0a0f1ab1e622bc9c1643d539d13219f921d977e262d0800d08db20d5781c6e2e \
  --frames-sha256 606e03fab3f7bba75700ab06d1e87b67bc7c7f0de8caac475319a695344f631b \
  --before-sha256 d6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355 \
  --after-sha256 6fe2540ab631aa557ebdf51e1f3fa70fa3395d7f86599585400d0fe5baa026aa
echo "DPA4C_PILOT_EVALUATION_COMPLETE"
CONTAINER
