# ELEMENTA large-scale CACE-SOG training

For the experimental DPA4C baseline and SOG comparison, see the separate
[DPA4C research workflow](README_DPA4C.md) and its outstanding validation and
data-loader limitations before starting an SCNet job.

For the current GPU-only SCNet access, use [Windows preparation and DPDispatcher
submission](docs/elementa_windows_scnet.md): prepare a complete disk index on
Windows, seal it, upload the immutable data and bundle once, bind the new paths
on SCNet, then dispatch GPU smoke/training/evaluation without re-uploading data.

See [the direct SCNet guide](docs/elementa_scnet.md) for the existing
`$HOME/envs/mlip` environment, direct commands, and the requirements for CPU
partition access. A visible GPU partition does not grant access to a CPU queue.

From this directory, use:

```bash
python -m cace_sog_training --help
```

The commands include `index`, `seal`, `bind`, `dispatch`, `environment`, `smoke`,
`converge`, `train`, and `evaluate`. Windows `index`/`seal` require only NumPy/ASE;
model runtime dependencies remain on the GPU machine. `dispatch` defaults to an
offline plan; `--submit` is required to connect and submit work. Defaults select
`sog_candidates`, `lattice_auxiliary`, and
`pbe_lowdim_auxiliary`, preserving existing `train/val/test` membership.
Indexing and per-frame graph construction are disk-backed; epoch metrics and
evaluation errors are accumulated in fixed-size counters. Every selected
training frame is used per epoch, including the last incomplete batch.

This first version supports one CPU or one GPU, E/F supervision, and three-
dimensionally periodic right-handed cells. It does not add spin inputs:
`spin_review` requires an explicit, reviewed consistent-state policy. Pending,
charge and boundary review channels are not automatic training inputs.

`best.pt` is selected using validation errors; `last.pt` stores model/optimizer,
training position, deterministic sampler, train-only elemental offsets, RNG,
and input identity for restart. Model weights are saved as `state_dict` plus
configuration rather than Python model objects. Existing examples and the
scientific kernel files are left unchanged.

Local CPU tests do not replace the requested SCNet A800 smoke, grid checks,
Molten-NaCl reproduction, or full ELEMENTA training.
