#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-restricted-resume
# Debug only: existing 16-frame train/val LMDBs; never reads test frames.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
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
RUN="$ROOT/runs/dpa4c-restricted-resume-$RESTRICTED_JOB"
test -d "$CODE/dpa4c_sog"
echo '64c0696636097e0d8f0817ba5f3df85d2610b2fd7ce7edc21d65714005641352  '"$CODE/dpa4c_sog/restricted_train_experimental.py" | sha256sum -c -
echo '70e3c63179d30c606b294ff23d6239b5b309818a064811d79fa04701825b4865  '"$CODE/dpa4c_sog/sampler_resume_experimental.py" | sha256sum -c -
echo '40a93ab3e359498edf94ed595a3aadd254f58d980c5ccc4581345e869f69d858  '"$CODE/dpa4c_sog/sampler_recovery.py" | sha256sum -c -
echo 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355  '"$INITIAL" | sha256sum -c -
mkdir "$RUN"
mkdir "$RUN/continuous" "$RUN/interrupted"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE"
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4
echo "RUN=$RUN"
python - "$CFG" "$DATA" <<'PY'
import json, sys
from pathlib import Path
config=json.loads(Path(sys.argv[1]).read_text()); data=Path(sys.argv[2])
t=config['training']; status=json.loads((data/'status.json').read_text())
if t['num_steps'] != 20 or t['save_freq'] != 1 or t['seed'] != 42:
    raise SystemExit('unexpected debug steps/save interval/seed')
if status['status'] != 'complete' or status['debug_only'] is not True or status['test_split_read'] is not False or status['counts'] != {'train':16,'val':16}:
    raise SystemExit('unexpected debug data status')
PY
cat > "$RUN/instrument.py" <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem
from deepmd.pt_expt.train.training import Trainer
from dpa4c_sog.sampler_recovery import DurableSamplerResume
from dpa4c_sog.restricted_train_experimental import main

log=Path(os.environ['RESTRICTED_LOG'])
def emit(row):
    with log.open('a',encoding='utf8') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
old_batch=LmdbDataSystem.get_batch
def batch(self,*a,**kw):
    result=old_batch(self,*a,**kw)
    source=Path(self._reader.lmdb_path).name
    if source not in ('train.lmdb','val.lmdb') or result.get('fid') is None:
        raise ValueError('unexpected split or missing FID')
    values=[int(x) for x in np.asarray(result['fid']).reshape(-1)]
    if len(values)!=1: raise ValueError('expected batch size 1')
    emit({'type':'batch','split':source[:-5],'fid':values[0]})
    return result
LmdbDataSystem.get_batch=batch
old_step=Trainer._optimizer_step
def step(self):
    before=[float(g['lr']) for g in self.optimizer.param_groups]
    old_step(self)
    after=[float(g['lr']) for g in self.optimizer.param_groups]
    emit({'type':'optimizer','lr_before':before,'lr_after':after})
Trainer._optimizer_step=step
old_checkpoint=DurableSamplerResume.checkpoint
def checkpoint(self,path,step):
    result=old_checkpoint(self,path,step)
    if os.environ.get('RESTRICTED_INTERRUPT')=='1' and step==16:
        emit({'type':'fault','step':step,'phase':'after_complete_pair'})
        os.kill(os.getpid(),9)
    return result
DurableSamplerResume.checkpoint=checkpoint
main()
PY
sha256sum "$RUN/instrument.py"
COMMON=(--config "$CFG" --train "$DATA/train.lmdb" --val "$DATA/val.lmdb" --initial "$INITIAL" --initial-sha256 d6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355)
RESTRICTED_LOG="$RUN/continuous.jsonl" RESTRICTED_INTERRUPT=0 python -u "$RUN/instrument.py" --mode start --run "$RUN/continuous" "${COMMON[@]}"
set +e
RESTRICTED_LOG="$RUN/interrupted.jsonl" RESTRICTED_INTERRUPT=1 python -u "$RUN/instrument.py" --mode start --run "$RUN/interrupted" "${COMMON[@]}"
RC=$?
set -e
[ "$RC" -eq 137 ] || { echo "Expected SIGKILL after step 16 pair, got $RC"; exit 1; }
test -s "$RUN/interrupted/model.ckpt-16.pt"
test -s "$RUN/interrupted/model.ckpt-16.pt.sampler.json"
test ! -e "$RUN/interrupted/restricted_complete.json"
RESTRICTED_LOG="$RUN/resumed.jsonl" RESTRICTED_INTERRUPT=0 python -u "$RUN/instrument.py" --mode resume --run "$RUN/interrupted" "${COMMON[@]}"
test -s "$RUN/continuous/restricted_complete.json"
test -s "$RUN/interrupted/restricted_complete.json"
python - "$RUN" "$DATA" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
root,data=map(Path,sys.argv[1:])
def records(name): return [json.loads(s) for s in (root/(name+'.jsonl')).read_text().splitlines()]
base, first, resumed=map(records,('continuous','interrupted','resumed'))
def batches(rows,split): return [x['fid'] for x in rows if x['type']=='batch' and x['split']==split]
side_path=root/'interrupted/model.ckpt-16.pt.sampler.json'
side=json.loads(side_path.read_text()); cp=root/'interrupted/model.ckpt-16.pt'
if side['step']!=16 or hashlib.sha256(cp.read_bytes()).hexdigest()!=side['checkpoint_sha256']:
    raise SystemExit('step16 checkpoint pair failed integrity check')
result={'status':'running','debug_only':True,'test_split_read':False,
        'checkpoint_step':16,'note':'single-rank batch=1; RNG state not restored; not production certification'}
for split in ('train','val'):
    full, pre, post=(batches(x,split) for x in (base,first,resumed))
    count=side['batch_counts'][split]
    if pre != full[:count] or post != full[count:]:
        raise SystemExit(split+' FID sequence mismatch')
    if split=='train':
        expected={int(json.loads(s)['key']) for s in (data/'train.jsonl').read_text().splitlines()}
        if len(expected)!=16 or len(set(full[:16]))!=16 or set(full[:16])!=expected:
            raise SystemExit('first epoch did not cover exact 16 source keys')
    result[split]={'checkpoint_batches':count,'initial_fids':pre,'expected_after':full[count:],'resumed':post}
def lrs(rows): return [x for x in rows if x['type']=='optimizer']
full,pre,post=map(lrs,(base,first,resumed))
if len(full)!=20 or len(pre)!=16 or len(post)!=4:
    raise SystemExit('unexpected optimizer step counts')
for i,(a,b) in enumerate(zip(full,pre+post)):
    for key in ('lr_before','lr_after'):
        if len(a[key])!=len(b[key]) or any(not math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-15) for x,y in zip(a[key],b[key])):
            raise SystemExit(f'learning rate mismatch at step {i+1} ({key})')
for name in ('continuous','interrupted'):
    complete=json.loads((root/name/'restricted_complete.json').read_text())
    if complete['status']!='complete' or complete['checkpoint']['step']!=20 or complete['rng_state_saved'] is not False:
        raise SystemExit('unexpected final checkpoint status: '+name)
result.update(status='passed',optimizer_steps=20,lr_sequences_match=True)
(root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
print('RESTRICTED_RESUME_COMPARISON_PASSED',flush=True)
PY
SCRIPT
