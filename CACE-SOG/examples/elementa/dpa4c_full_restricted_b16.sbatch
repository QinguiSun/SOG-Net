#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=dpa4c-full-b16
# Usage: sbatch ... this-file start|resume PREPARED RUN REPOSITORY_COMMIT
# Long run: status, validation and final test remain separate operations.
set -euo pipefail
MODE=${1:?start or resume required}
PREPARED=${2:?prepared directory required}
RUN=${3:?run directory required}
EXPECTED_COMMIT=${4:?full repository commit required}
[ "$#" -eq 4 ] || { echo 'Expected exactly four arguments'; exit 2; }
case "$MODE" in start|resume) ;; *) echo 'invalid mode'; exit 2;; esac
case "$PREPARED" in /work/*) ;; *) echo 'prepared must be under /work'; exit 2;; esac
case "$RUN" in /work/*) ;; *) echo 'run must be under /work'; exit 2;; esac
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-b8-9a575
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
CODE="$ROOT/replay-experiment-20261102/restricted-code-b16"
ACTUAL_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
[ "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" ] || { echo 'repository commit mismatch'; exit 1; }
(cd "$REPO" && git diff --quiet && git diff --cached --quiet) || { echo 'repository tracked changes'; exit 1; }
if [ "$MODE" = start ]; then mkdir "$RUN"; else test -d "$RUN"; fi
echo 'b66245b6b8d1fca4638a37d8ab388625cbf48ade32d0d346730d9c5f627c635e  '"$CODE/dpa4c_sog/restricted_train_batch16_experimental.py" | sha256sum -c -
echo '70e3c63179d30c606b294ff23d6239b5b309818a064811d79fa04701825b4865  '"$CODE/dpa4c_sog/sampler_resume_experimental.py" | sha256sum -c -
echo '40a93ab3e359498edf94ed595a3aadd254f58d980c5ccc4581345e869f69d858  '"$CODE/dpa4c_sog/sampler_recovery.py" | sha256sum -c -
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?GPU missing}"
export APPTAINERENV_FULL_MODE="$MODE" APPTAINERENV_FULL_PREPARED="$PREPARED" APPTAINERENV_FULL_RUN="$RUN" APPTAINERENV_FULL_COMMIT="$ACTUAL_COMMIT"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOT/ubuntu-jammy-rootfs-5414994" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-b8-9a575/CACE-SOG
CODE="$ROOT/replay-experiment-20261102/restricted-code-b16"
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
RELEASE="$ROOT/DPA4C-Neo-OMol25-100M-v20260820.json"
RAW="$ROOT/DPA4C-Neo-OMol25-100M-v20260820.pt"
INITIAL="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
DATA="$ROOT/train-val-lmdb-full-5417356"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" OMP_NUM_THREADS=4
test -r /.singularity.d/libs/libcuda.so.1
cd "$REPO"
export PYTHONPATH="$ROOT/ase-support-5416043:$REPO"
python - "$FULL_PREPARED" "$INDEX" "$RELEASE" "$RAW" "$INITIAL" "$FULL_COMMIT" "$DATA" <<'PY'
import json, sys
from pathlib import Path
from dpa4c_sog.train_run_guard import checked_files
prepared,index,release,raw,initial,commit,data=sys.argv[1:]
checked=checked_files(prepared,index,release,raw,commit,initialization='pilot-calibrated',calibrated=initial)
if checked['source_counts'] != {'train':3543786,'val':443734} or checked['source_index_identity'] != 'c07665508e5499fa61dcb5d2faa06ef81fdce40da68f0a88a308ecaf24795251':
    raise ValueError('unexpected full index identity/counts')
c=json.loads((Path(prepared)/'input_dpa4c_lmdb.json').read_text()); t=c['training']
if (t.get('num_steps') != 221493 or t.get('seed') != 42 or t.get('save_freq') != 5000 or t.get('max_ckpt_keep',0) < 100
    or t['training_data'].get('batch_size') != 16 or t['training_data'].get('auto_prob_style') != 'prob_sys_size'
    or t['validation_data'].get('batch_size') != 1 or t['validation_data'].get('numb_batch') != 1 or t.get('change_bias_after_training') is not False):
    raise ValueError('configuration does not match fixed full-run schedule')
for name, split in (('training_data','train'),('validation_data','val')):
    if Path(t[name]['systems']).resolve() != (Path(data)/(split+'.lmdb')).resolve(): raise ValueError('unexpected dataset')
coverage_path=Path('/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c/runs/dpa4c-sampler-b16-coverage-5418876/coverage.json')
coverage=json.loads(coverage_path.read_text())
if (coverage.get('status') != 'passed' or coverage.get('frames') != 3543786 or coverage.get('batches') != 221493
    or coverage.get('batch_size') != 16 or coverage.get('seed') != 42 or coverage.get('duplicates') != 0
    or coverage.get('missing') != 0 or coverage.get('out_of_range') != 0
    or coverage.get('train_lmdb_sha256') != checked['converted_hashes']['train_lmdb_sha256']):
    raise ValueError('full batch16 coverage result mismatch')
resume_path=Path('/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c/runs/dpa4c-batch16-resume-5418877/comparison.json')
resume=json.loads(resume_path.read_text())
if (resume.get('status') != 'passed' or resume.get('debug_only') is not True
        or resume.get('test_split_read') is not False or resume.get('checkpoint_step') != 3
        or resume.get('optimizer_steps') != 6 or resume.get('lr_sequences_match') is not True):
    raise ValueError('batch16 restart comparison result mismatch')
print('FULL_INDEX_CONFIG_AND_HASH_CHECK_OK; val minibatch diagnostic only; RNG not restored',flush=True)
PY
export PYTHONPATH="$CODE"
echo 'dfa3c7a621e98916913e4bf8bdbfb3b09522a6daef77a097797b3aecfc47322c  '"$CODE/dpa4c_sog/warmstart_provenance.py" | sha256sum -c -
python -u -m dpa4c_sog.warmstart_provenance --run "$ROOT/runs/dpa4c-full-b8-v1" --prepared "$ROOT/prepared-full-b8-v1" --conversion "$DATA" --checkpoint-sha256 70ac1cf73ad28a383e633cc9b4b48478e1eb3408221bb4a9225fceecad7f81ea --fingerprint e5b00d12b1157a1766acfa9c41a8c1c5478db065d1f7995d1d2cc380c5c0e1d4
WARM="$ROOT/runs/dpa4c-full-b8-v1/model.ckpt-245000.pt"
cd "$FULL_RUN"
python -u -m dpa4c_sog.restricted_train_batch16_experimental --mode "$FULL_MODE" --config "$FULL_PREPARED/input_dpa4c_lmdb.json" --run "$FULL_RUN" --train "$DATA/train.lmdb" --val "$DATA/val.lmdb" --initial "$WARM" --initial-sha256 70ac1cf73ad28a383e633cc9b4b48478e1eb3408221bb4a9225fceecad7f81ea
echo FULL_TRAIN_COMMAND_FINISHED
CONTAINER
