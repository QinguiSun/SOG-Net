#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dpa4c-synthetic-plot
# Synthetic input and fabricated test-like status only. No real train/val/test data.
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
CODE="$ROOT/replay-experiment-20261102/val-code"
PYTHON="$HOME/envs/mlip/bin/python"
RUN="$ROOT/runs/SYNTHETIC-ONLY-plot-smoke-${SLURM_JOB_ID:?}"
test -x "$PYTHON"
test -r "$CODE/dpa4c_sog/baseline_metrics.py"
mkdir "$RUN"
export PYTHONPATH="$CODE"
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$RUN/mpl-config"
export MPLBACKEND=Agg
echo "SYNTHETIC ONLY: artificial numbers, no real test data. Directory: $RUN"
"$PYTHON" - "$RUN" <<'PY'
import hashlib, json, sys
from pathlib import Path
import numpy, matplotlib
run = Path(sys.argv[1])
print('Python:', sys.executable, 'NumPy:', numpy.__version__, 'Matplotlib:', matplotlib.__version__, flush=True)
(run / 'SYNTHETIC_ONLY_NOT_REAL_TEST.txt').write_text('Every input, output, and figure in this directory is synthetic. No model was run and no actual ELEMENTA frame was read. The plot title is the production default and is not a true held-out test.\n')
rows = []
for i, (e, ep, ef) in enumerate([(-4.0, -2.0, -3.8), (-2.0, -4.0, -2.4)]):
    ref = [[float(i+j+k)/3 for k in range(3)] for j in range(2)]
    rows.append({'split':'test', 'frame_key':'SYNTHETIC_FRAME_'+str(i), 'natoms':2,
      'reference_energy_eV':e, 'pretrained_energy_eV':ep, 'finetuned_energy_eV':ef,
      'reference_forces_eV_A':ref,
      'pretrained_forces_eV_A':[[x+0.2 for x in atom] for atom in ref],
      'finetuned_forces_eV_A':[[x-0.1 for x in atom] for atom in ref]})
path=run/'SYNTHETIC-ONLY.jsonl'
with path.open('x',encoding='utf8') as f:
    for row in rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
status={'status':'complete','is_shard':False,'ordinal_start':0,'ordinal_end':2,
 'test_total_frames':2,'expected_frames':2,'frames_written':2,'split':'test',
 'channel':'sog_candidates','predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'SYNTHETIC_ONLY':True,'WARNING':'Fabricated records and status; no real test data.'}
Path(str(path)+'.status.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
PY
"$PYTHON" -m dpa4c_sog.baseline_metrics --input "$RUN/SYNTHETIC-ONLY.jsonl" --output "$RUN/SYNTHETIC-ONLY-result" --bins 32
"$PYTHON" - "$RUN/SYNTHETIC-ONLY-result" <<'PY'
import json, math, sys
from pathlib import Path
root=Path(sys.argv[1]); result=json.loads((root/'metrics.json').read_text(encoding='utf8'))
assert result['status']=='complete' and result['paired_frames']==2
assert result['prediction_provenance']['status']['SYNTHETIC_ONLY'] is True
metrics=result['metrics']
for model, prop, expected in [('pretrained','energy',1.0),('finetuned','energy',0.15),('pretrained','force',0.2),('finetuned','force',0.1)]:
    actual=metrics[model][prop]['mae']
    assert math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12), (model,prop,actual,expected)
for model in ('pretrained','finetuned'):
    path=root/(model+'_parity.png')
    with path.open('rb') as f: assert f.read(8)==b'\x89PNG\r\n\x1a\n'
    assert path.stat().st_size>1000
print('SYNTHETIC_ONLY: expected MAEs and both PNG files verified')
print('SYNTHETIC_PLOT_SMOKE_PASSED')
PY
