#!/usr/bin/env bash
# CPU-only index construction; supply the allowed --time when submitting.
# Create CACE-SOG/logs before submission. No GPU is requested by this job.
#SBATCH --job-name=elementa-index
#SBATCH --partition=qdhcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/elementa-index-%j.out
#SBATCH --error=logs/elementa-index-%j.err
set -euo pipefail

project_dir="${PROJECT_DIR:-/work/home/einlovestan/projects/SOG-Net/CACE-SOG}"
cd "$project_dir"
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  conda_base="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
else
  echo "Conda is not on this batch job's PATH. Load the Conda module used in your verified mlip session, then resubmit." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"
conda activate "${MLIP_ENV:-$HOME/envs/mlip}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
date -u
git rev-parse HEAD
python -c 'import sys; print(sys.executable)'
bash examples/elementa/run_single_gpu.sh index
date -u
