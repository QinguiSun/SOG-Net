#!/bin/bash
#SBATCH --partition=qdagnormal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=dpa4c-lmdb-recover

set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ROOTFS="$ROOT/ubuntu-jammy-rootfs-5414994"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune
test -d "$ROOTFS"
test -f "$REPO/CACE-SOG/dpa4c_sog/index_lmdb_resume.py"
echo "NODE=$(hostname)"
echo "REPOSITORY_COMMIT=$(cd "$REPO" && git rev-parse HEAD)"
echo "RECOVERY_CONVERTER_SHA256=$(sha256sum "$REPO/CACE-SOG/dpa4c_sog/index_lmdb_resume.py")"
module load apps/apptainer/1.3.4
export APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?No Slurm GPU assignment}"
export APPTAINERENV_DPA_RECOVER_JOB="$SLURM_JOB_ID"
apptainer exec --underlay --nv --cleanenv --no-home --bind /work:/work "$ROOTFS" /bin/bash -s <<'CONTAINER'
set -euo pipefail
ROOT=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/dpa4c
ENV_DIR="$ROOT/env-deepmd-3.2.0-cuda129-retry-5415731"
REPO=/work/home/einlovestan/projects/SOG-Net-dpa4c-finetune/CACE-SOG
INDEX=/work/home/einlovestan/datasets/ELEMENTA_CACE_SOG/indices/elementa.sqlite3
RUN="$ROOT/runs/dpa4c-lmdb-recover-$DPA_RECOVER_JOB"
set +u
source "$ENV_DIR/bin/activate" "$ENV_DIR"
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/.singularity.d/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/ase-support-5416043:$REPO"
export OMP_NUM_THREADS=2
mkdir "$RUN"
echo "RUN_DIRECTORY=$RUN"
cd "$REPO"
set +e
python -u - "$INDEX" "$RUN/data" <<'PY'
import os, sys
from dpa4c_sog.index_lmdb_resume import convert
def kill(stage, split, count):
    if stage == 'after_manifest_fsync' and split == 'train' and count == 4:
        print('INTENTIONAL_PROCESS_EXIT_BEFORE_COMMIT', flush=True)
        os._exit(71)
convert(sys.argv[1], sys.argv[2], commit_every=4, limit_per_split=16, fault=kill)
PY
rc=$?
set -e
if [ "$rc" -ne 71 ]; then echo "Fault injection did not stop at expected point: $rc"; exit 1; fi
python -u -m dpa4c_sog.index_lmdb_resume --database "$INDEX" --output "$RUN/data" --limit-per-split 16 --commit-every 4
python - "$INDEX" "$RUN/data" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from ase.data import chemical_symbols
from cace_sog_training.data import IndexReader
from deepmd.dpmodel.utils.lmdb_data import LmdbDataReader

data = Path(sys.argv[2])
status = json.loads((data / 'status.json').read_text())
assert status['status'] == 'complete' and status['debug_only'] and not status['test_split_read']
with IndexReader(sys.argv[1]) as source:
    for split in ('train', 'val'):
        lmdb = LmdbDataReader(str(data / (split + '.lmdb')), list(chemical_symbols[1:119]), batch_size=1)
        assert len(lmdb) == 16
        rows = [json.loads(s) for s in (data / (split + '.jsonl')).open()]
        for i, row in enumerate(rows):
            rec = source.record('sog_candidates', split, row['source_ordinal'])
            assert rec['id'] == row['source_id'] and rec['sha256'] == row['frame_sha256']
            atoms = source.read_atoms(rec)
            order = np.asarray(row['canonical_to_source'])
            frame = lmdb[i]
            np.testing.assert_allclose(frame['coord'], atoms.positions[order], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(frame['box'].reshape(3,3), atoms.cell.array, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(frame['force'], atoms.arrays['REF_forces'][order], rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(frame['energy'], [atoms.info['REF_energy']], rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(frame['atype'], atoms.numbers[order] - 1)
        lmdb.close()
print('LMDB_SCNET_CRASH_RECOVERY_AND_SOURCE_MATCH_OK')
PY
CONTAINER
