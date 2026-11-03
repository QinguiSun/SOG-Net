#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpa4c-batch-probe
# Debug only: no actual test frames; all checkpoint outputs are diagnostic.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?GPU missing}"
export APPTAINERENV_BATCH_PROBE_JOB="${SLURM_JOB_ID:?Job ID missing}"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOT/ubuntu-jammy-rootfs-5414994" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
CODE="$ROOT/replay-experiment-20261102/restricted-code-v2"
DATA="$ROOT/train-val-lmdb-full-5417356"
SOURCE="$ROOT/prepared-full-epoch1-v1/input_dpa4c_lmdb.json"
INITIAL="$ROOT/runs/dpa4c-bias-pilot-5416085/pretrained-pilot-bias.pt"
RUN="$ROOT/runs/dpa4c-batch-probe-$BATCH_PROBE_JOB"
test -r "$SOURCE"
echo '74b6c95591a1aa2292b55a7c5822d8c48af8f9b1b0002454e50cc7508f477177  '"$CODE/dpa4c_sog/restricted_train_experimental.py" | sha256sum -c -
echo 'd6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355  '"$INITIAL" | sha256sum -c -
mkdir "$RUN"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" OMP_NUM_THREADS=4
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "RUN=$RUN; DEBUG ONLY: weights must not be used for production"
nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv -l 2 > "$RUN/gpu-monitor.csv" 2>&1 &
MONITOR=$!
trap 'kill "$MONITOR" 2>/dev/null || true' EXIT
python - "$SOURCE" "$DATA" "$RUN" <<'PY'
import json,sys
from pathlib import Path
source,data,run=map(Path,sys.argv[1:]); cfg=json.loads(source.read_text()); t=cfg['training']
status=json.loads((data/'status.json').read_text())
assert status['status']=='complete' and status['no_sampling'] is True and status['test_split_read'] is False
assert status['counts']=={'train':3543786,'val':443734}
assert Path(t['training_data']['systems']).resolve()==(data/'train.lmdb').resolve()
assert Path(t['validation_data']['systems']).resolve()==(data/'val.lmdb').resolve()
for b in (8,16,32):
    config=json.loads(json.dumps(cfg)); z=config['training']
    z['training_data']['batch_size']=b; z['validation_data']['batch_size']=1
    z.update(num_steps=16,save_freq=16,max_ckpt_keep=2,disp_freq=16)
    (run/f'input-b{b}.json').write_text(json.dumps(config,indent=2)+'\n')
    (run/f'b{b}').mkdir()
PY
cat > "$RUN/driver.py" <<'PY'
import copy,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from deepmd.pt_expt.train.training import Trainer
from deepmd.pt_expt.utils.lmdb_dataset import LmdbDataSystem
import dpa4c_sog.restricted_train_experimental as entry
b=int(os.environ['PROBE_BATCH']); assert b in (8,16,32)
assert torch.cuda.is_available(); old_validate=entry.validate_config
def debug_validate(cfg, train, val):
    assert cfg['training']['training_data']['batch_size']==b
    assert cfg['training']['validation_data']['batch_size']==1
    check=copy.deepcopy(cfg); check['training']['training_data']['batch_size']=1
    old_validate(check,train,val)
entry.validate_config=debug_validate
old_step=Trainer.train_step; old_batch=LmdbDataSystem.get_batch
times=[]; frame_counts=[]
def batch(system,*a,**kw):
    result=old_batch(system,*a,**kw)
    if Path(system._reader.lmdb_path).name=='train.lmdb':
        fid=result.get('fid')
        if fid is None: raise ValueError('missing train FID')
        frame_counts.append(len(np.asarray(fid).reshape(-1)))
    return result
def step(trainer,*a,**kw):
    torch.cuda.synchronize(); t=time.monotonic()
    result=old_step(trainer,*a,**kw)
    torch.cuda.synchronize(); times.append(time.monotonic()-t)
    return result
Trainer.train_step=step; LmdbDataSystem.get_batch=batch
torch.cuda.reset_peak_memory_stats(); start=time.monotonic()
entry.main()
if len(times)!=16 or len(frame_counts)!=16: raise ValueError('unexpected train steps or counts')
warm=3; elapsed=sum(times[warm:]); frames=sum(frame_counts[warm:])
data={'status':'complete','debug_only':True,'test_split_read':False,'batch_size':b,
      'training_steps':len(times),'frame_counts':frame_counts,'warm_steps_discarded':warm,
      'warm_train_step_seconds':elapsed,'warm_frames_per_second':frames/elapsed,
      'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
      'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
      'limitation':'16 steps of full LMDB; warm timings exclude validation/checkpoint overhead; not production weights'}
Path(os.environ['PROBE_RESULT']).write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2),flush=True)
PY
for b in 8 16 32; do
  echo "=== BATCH $b ==="
  set +e
  PROBE_BATCH="$b" PROBE_RESULT="$RUN/result-b$b.json" python -u "$RUN/driver.py" --mode start --config "$RUN/input-b$b.json" --run "$RUN/b$b" --train "$DATA/train.lmdb" --val "$DATA/val.lmdb" --initial "$INITIAL" --initial-sha256 d6b95b6665dc9bf4b15f4f0c93ca4f6b2d9029b700a42ffcc65502f3cdda2355
  rc=$?
  set -e
  echo "BATCH_EXIT b=$b code=$rc"
done
echo "PROBE_RUN_DIRECTORY=$RUN"
if [ ! -s "$RUN/result-b8.json" ] && [ ! -s "$RUN/result-b16.json" ] && [ ! -s "$RUN/result-b32.json" ]; then
  echo 'No batch configuration completed; inspect the log.'
  exit 1
fi
echo BATCH_TRAINING_PROBE_FINISHED
CONTAINER
