#!/usr/bin/env bash
# Supply the real --account and --time when submitting.
# Create CACE-SOG/logs before sbatch opens the output files.
#SBATCH --job-name=elementa-sog
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/elementa-%j.out
#SBATCH --error=logs/elementa-%j.err
set -euo pipefail

project_dir="${PROJECT_DIR:-/work/home/einlovestan/projects/SOG-Net/CACE-SOG}"
cd "$project_dir"

if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  conda_base="$("$CONDA_EXE" info --base)"
elif command -v conda >/dev/null 2>&1; then
  conda_base="$(conda info --base)"
else
  echo "Conda is not on this batch job's PATH. Load the same Conda module used in your verified mlip session, then submit again." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$conda_base/etc/profile.d/conda.sh"
conda activate "${MLIP_ENV:-$HOME/envs/mlip}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
date -u
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi; fi
git rev-parse HEAD
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "No CUDA device is visible in the allocated job"'
bash examples/elementa/run_single_gpu.sh "${STAGE:-train}"
date -u
