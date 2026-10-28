#!/usr/bin/env bash
#SBATCH --partition=qdagnormal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
set -euo pipefail

# Supply absolute paths via the submission environment. A NEW run directory is
# required; this script never silently starts over in an existing run.
: "${DPA_ENV:?set DPA_ENV to the separately verified DeePMD Conda prefix}"
: "${DPA_INPUT:?set DPA_INPUT to prepared input_dpa4c.json}"
: "${DPA_PRETRAINED:?set DPA_PRETRAINED to the matching OMol checkpoint}"
: "${DPA_RUN:?set DPA_RUN to a new persistent output directory}"

if [[ -n "${DPA_CONDA_SH:-}" ]]; then
  source "$DPA_CONDA_SH"
elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
  DPA_CONDA_BASE="$("$CONDA_EXE" info --base)"
  source "$DPA_CONDA_BASE/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  DPA_CONDA_BASE="$(conda info --base)"
  source "$DPA_CONDA_BASE/etc/profile.d/conda.sh"
else
  printf '%s\n' 'Conda initialization not found; set DPA_CONDA_SH to the verified conda.sh path.' >&2
  exit 2
fi
conda activate "$DPA_ENV"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONNOUSERSITE=1
python -c 'import deepmd, torch; print("deepmd", deepmd.__version__, "torch", torch.__version__, "CUDA", torch.version.cuda); assert torch.cuda.is_available(), "CUDA GPU is unavailable"'
command -v dp >/dev/null
test -f "$DPA_INPUT"
test -f "$DPA_PRETRAINED"
python - "$DPA_INPUT" <<'PY'
import json, sys
from pathlib import Path
cfg = json.loads(Path(sys.argv[1]).read_text())
model = cfg['model']
if model.get('descriptor', {}).get('type') != 'dpa4c' or model.get('long_range'):
    raise SystemExit('This entry point is for the short-range DPA4C baseline only.')
if cfg.get('validating', {}).get('full_validation'):
    raise SystemExit('Use the external streaming full-validation command for this large dataset.')
PY
mkdir -p "$(dirname -- "$DPA_RUN")"
mkdir "$DPA_RUN"
cd "$DPA_RUN"
dp --pt-expt train "$DPA_INPUT" --finetune "$DPA_PRETRAINED"
