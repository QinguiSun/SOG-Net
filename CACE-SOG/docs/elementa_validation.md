# Initial adapter validation

Base repository revision: `b249fc9bf1e92f5a0d4a1432acd26d7f4f0db89f`.

Local validation used owned, tiny periodic fixtures, not the user's NAS or SCNet
data. All 37 CPU tests passed, including 24 parameterized/subtest cases in the
reported test run. The full terminal sequence was exercised as subprocesses:
partial index → resume index → smoke → partial training → resume training →
validation-best checkpoint → per-channel test evaluation → trained-grid check.

Validated behavior includes:

- Raw SHA, E/F labels, original splits, component leakage rejection, unchanged
  input verification, transaction checkpoint recovery and partial-index guards.
- Full total-force backward through CACE plus periodic SOG; Adam updates to the
  short-range, latent and kernel parameters; finite-difference force checks.
- Agreement with the upstream periodic kernel for energies, forces and kernel/q
  gradients, with both self-interaction settings.
- Strict state_dict/config reload; CPU interrupted/resumed training matches the
  uninterrupted result exactly for the tested deterministic fixture.
- Every selected training frame, including the final incomplete batch, is used;
  train-only elemental offsets; validation-selected best.pt; separate test
  metrics for all selected channels.
- Fixed-size metric counters and O(1)-state sampler; metadata/reference-policy
  validation; explicit Spin opt-in and single-writer run protection.

The additional default-model smoke used two periodic NaCl fixtures for ten
optimizer steps. It updated 21 parameters, had finite nonzero wl/sl gradients,
and reloaded energy/force predictions with zero difference. Its training loss
changed from approximately 2.59165 before the first step to 2.22912 before the
last step. Process peak RSS was about 408 MB; this small fixed-batch test is not
a memory or throughput guarantee for the full ELEMENTA dataset.

Local environment: macOS arm64, Python 3.12.14, PyTorch 2.14.1 (CPU), NumPy 2.5.4,
ASE 3.29.0, matscipy 1.3.1, pytorch-finufft 0.1.0, Lightning 2.6.6. The upstream
code emits torch.jit and ASE/NumPy deprecation warnings in this environment.

The GitHub workflow runs the same CPU tests on Python 3.10/Linux. Its actual
status is reported by GitHub Actions; a workflow file is not evidence that its
remote run has passed.

Not run here: native Windows/WinSCP/SFTP transfer, SCNet Slurm submission, the
adapted A800 GPU path, a complete Molten-NaCl reproduction, or multi-million-frame
ELEMENTA training. The user's earlier A800 tests apply to the original main
revision, and are not claimed as tests of this adapter. Follow the SCNet guide
to perform the new smoke, numerical and memory checks in the existing mlip env.
