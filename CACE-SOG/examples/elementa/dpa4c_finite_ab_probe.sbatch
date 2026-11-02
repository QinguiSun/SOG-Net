#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-finite-ab
# Diagnostics only: never use arm B weights in production; no test frames.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?GPU missing}"
export APPTAINERENV_PROBE_JOB="${SLURM_JOB_ID:?Job ID missing}"
export APPTAINERENV_EXPECTED_RESTRICTED_SHA="${EXPECTED_RESTRICTED_SHA:?set expected restricted module SHA}"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOT/ubuntu-jammy-rootfs-5414994" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
SOURCE_CODE="$ROOT/replay-experiment-20261102/restricted-code-v2/dpa4c_sog"
SOURCE_CONFIG="$ROOT/runs/dpa4c-epoch-fault-probe-5417774/input.json"
DATA="$ROOT/runs/dpa4c-lmdb-recover-5416703/data"
INITIAL="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-finite-ab-$PROBE_JOB"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
test -r /.singularity.d/libs/libcuda.so.1
echo "$EXPECTED_RESTRICTED_SHA  $SOURCE_CODE/restricted_train_experimental.py" | sha256sum -c -
echo 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355  '"$INITIAL" | sha256sum -c -
mkdir "$RUN"
mkdir -p "$RUN/code/dpa4c_sog" "$RUN/A" "$RUN/B"
cp "$SOURCE_CODE/restricted_train_experimental.py" "$SOURCE_CODE/sampler_resume_experimental.py" "$SOURCE_CODE/sampler_recovery.py" "$RUN/code/dpa4c_sog/"
touch "$RUN/code/dpa4c_sog/__init__.py"
echo "$EXPECTED_RESTRICTED_SHA  $RUN/code/dpa4c_sog/restricted_train_experimental.py" | sha256sum -c -
export PYTHONPATH="$RUN/code"
python - "$SOURCE_CONFIG" "$DATA" "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
source,data,run=map(Path,sys.argv[1:]); status=json.loads((data/'status.json').read_text())
assert status['status']=='complete' and status['debug_only'] is True and status['test_split_read'] is False
assert status['counts']=={'train':16,'val':16}
for split in ('train','val'):
    h=hashlib.sha256((data/(split+'.lmdb')/'data.mdb').read_bytes()).hexdigest()
    assert h==status[split+'_lmdb_sha256']
cfg=json.loads(source.read_text()); t=cfg['training']
assert t['seed']==42 and t['training_data']['auto_prob_style']=='prob_sys_size'
assert Path(t['training_data']['systems']).resolve()==(data/'train.lmdb').resolve()
assert Path(t['validation_data']['systems']).resolve()==(data/'val.lmdb').resolve()
t.update(num_steps=40, save_freq=10, max_ckpt_keep=10, disp_freq=10)
p=run/'benchmark-input.json'; p.write_text(json.dumps(cfg,sort_keys=True,indent=2)+'\n')
print('CONFIG_SHA256='+hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
PY
cat > "$RUN/driver.py" <<'PY'
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from deepmd.pt_expt.train.training import Trainer
from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem
import dpa4c_sog.restricted_train_experimental as module
mode=os.environ['BENCH_ARM']; output=Path(os.environ['BENCH_OUTPUT'])
assert mode in ('A','B') and torch.cuda.is_available()
finite=module.finite_step; old_step=Trainer.train_step; old_batch=LmdbDataSystem.get_batch
optim_times=[]; step_times=[]; batches=[]
def bench_finite(original, trainer, torch_module):
    torch.cuda.synchronize(); start=time.monotonic()
    if mode=='A': finite(original,trainer,torch_module)
    else: original(trainer)
    torch.cuda.synchronize(); optim_times.append(time.monotonic()-start)
def timed_step(trainer,*args,**kwargs):
    torch.cuda.synchronize(); start=time.monotonic()
    result=old_step(trainer,*args,**kwargs)
    torch.cuda.synchronize(); step_times.append(time.monotonic()-start)
    return result
def record_batch(system,*args,**kwargs):
    result=old_batch(system,*args,**kwargs)
    split=Path(system._reader.lmdb_path).name
    if split not in ('train.lmdb','val.lmdb') or result.get('fid') is None: raise ValueError('bad batch')
    batches.append([split,[int(x) for x in np.asarray(result['fid']).reshape(-1)]])
    return result
module.finite_step=bench_finite
Trainer.train_step=timed_step
LmdbDataSystem.get_batch=record_batch
start=time.monotonic()
module.main()
elapsed=time.monotonic()-start
if len(optim_times)!=40 or len(step_times)!=40: raise ValueError('unexpected number of steps')
data={'arm':mode,'debug_only':True,'test_split_read':False,'do_not_use_weights':True,
      'elapsed_seconds':elapsed,'optimizer_seconds':optim_times,'train_step_seconds':step_times,
      'batches':batches,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
      'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
PY
echo 'DIAGNOSTIC ONLY: arm B disables finite-step checks; its weights must never be used.'
COMMON=(--mode start --config "$RUN/benchmark-input.json" --train "$DATA/train.lmdb" --val "$DATA/val.lmdb" --initial "$INITIAL" --initial-sha256 d6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355)
for ARM in A B; do
  BENCH_ARM="$ARM" BENCH_OUTPUT="$RUN/$ARM.json" python -u "$RUN/driver.py" "${COMMON[@]}" --run "$RUN/$ARM"
done
python - "$RUN" <<'PY'
import json, statistics, sys
from pathlib import Path
r=Path(sys.argv[1]); a,b=[json.loads((r/(k+'.json')).read_text()) for k in ('A','B')]
if a['batches']!=b['batches']: raise SystemExit('different batch identity sequences')
out={'status':'complete','debug_only':True,'test_split_read':False,'same_batch_ids':True,
     'note':'40 steps on 16 frames; A then B in one job; exclude first 5 steps. Hardware/kernel warmup and shared node may bias results.'}
for obj in (a,b):
    out[obj['arm']]={'elapsed_seconds':obj['elapsed_seconds'],
      'optimizer_mean_s':statistics.mean(obj['optimizer_seconds'][5:]),
      'optimizer_median_s':statistics.median(obj['optimizer_seconds'][5:]),
      'train_step_mean_s':statistics.mean(obj['train_step_seconds'][5:]),
      'train_step_median_s':statistics.median(obj['train_step_seconds'][5:]),
      'peak_allocated_bytes':obj['peak_allocated_bytes'],'peak_reserved_bytes':obj['peak_reserved_bytes']}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2),flush=True)
print('FINITE_AB_DIAGNOSTIC_COMPLETE',flush=True)
PY
CONTAINER
