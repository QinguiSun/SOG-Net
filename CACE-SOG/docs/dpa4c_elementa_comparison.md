# DPA4C fine-tuning and comparison on ELEMENTA

This is an experimental, DPA4C-only workflow. The original test partition is
held out: train on `sog_candidates/train`, select checkpoints on
`sog_candidates/val`, then compare the untouched released checkpoint against
the selected checkpoint on the same complete `sog_candidates/test` frames.
The released checkpoint is not aligned or calibrated before its test result.
Fine-tuning may adjust its own energy bias from train labels; never fit that
bias to val or test. The constant [0, 1] charge and spin multiplicity is a
model-input assumption, not an electronic-state label measured in ELEMENTA.

## Readiness

The LMDB converter and batch jobs below are experimental. The pilot exhibits
very large force errors. First run the LMDB smoke on the actual installed
DeepMD build, verify labels, training and checkpoint reload, and measure RAM,
GPU memory and throughput at increasing scales. LMDB metadata and samplers
scale with frame count. Do not start full conversion or production training
merely because the Python tests pass. The production training launcher,
robust bias initialization and full-scale memory envelope remain to be validated.

`index_lmdb` defaults to every train and val frame, no test; `--limit-per-split`
is only for debug and marks the output accordingly. It uses a disk-backed
record store, but metadata memory is O(frames), capped by
`--max-frames-per-split`. All frames use one logical source system: sampling
weights are frame-level, not parent-material-balanced. Map size and free disk
space need an explicit check before full conversion. The original `index_lmdb`
converter cannot resume after interruption. An independent experimental
`index_lmdb_resume` converter provides transaction-based recovery, tested with
process termination on local LMDB; it still needs a kill/restart test on the
SCNet filesystem and cannot resume old converter output. Never treat partial
conversion status as complete.

On SCNet, `examples/elementa/dpa4c_lmdb_probe.sbatch` inspects the installed
environment. `examples/elementa/dpa4c_lmdb_smoke.sbatch` builds a 16-frame
per split debugging dataset and runs 10 steps. Both are configured for the
known SCNet paths and GPU partition; the smoke output is not a trained model.
`dpa4c_lmdb_scale.sbatch` benchmarks 10,000 debug frames per split.
`dpa4c_lmdb_resume_check.sbatch` probes interruption recovery on the SCNet
filesystem and checks decoded frames against the original index.

`prepare_lmdb` generates a config only after a complete unsampled conversion
passes source and hash checks. `dpa4c_train_lmdb.sbatch` provides guarded
start/resume entry points and checks the released checkpoint, input hashes,
and repository commit before calling DeePMD. This launcher has local tests
but has not yet been verified on SCNet. The user must explicitly set the
number of optimizer steps; a fixed number does not prove every frame is seen.

## Held-out evaluation after a production fine-tune exists

Use `dpa4c_sog.validate --kind dpa` to select among candidate checkpoints
using the entire original validation split; the report and checkpoint must
match the bound index. Each test `baseline_predict` invocation takes that
report, the selected checkpoint, the *original* released checkpoint, and
explicit `[0, 1]`. Use `--start` and `--end` for disjoint half-open ordinal
ranges, keeping the status sidecar for each completed shard. A failed shard
must be rerun to a new output path. `baseline_merge` verifies hashes, matching
model/index/validation provenance, ordering, no overlaps and exact full-test
coverage; it rejects incomplete shards. `baseline_metrics` accepts only a
complete merged prediction with a matching completion sidecar and writes
`metrics.json`, `pretrained_parity.png`, and `finetuned_parity.png`.

Metrics use energy per atom in eV/atom, all Cartesian force components in
eV/angstrom, and mean difference = prediction minus DFT. Parity figures use
all values in two-dimensional density histograms, rather than plotting a
random sample. Original large reference-energy offsets can dominate the
raw pretrained energy plot; optional calibrated comparisons must be separately
labeled and are not a substitute for the raw requested baseline. The plots
and metrics do not establish long-range causal accuracy.

For CLI syntax run `python -m dpa4c_sog.<module> --help` for `validate`,
`baseline_predict`, `baseline_merge`, and `baseline_metrics`. SCNet end-to-end
inference and plotting have not yet been established; Matplotlib is required
for the latter.
