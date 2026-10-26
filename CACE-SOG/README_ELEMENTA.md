# ELEMENTA large-scale CACE-SOG training

See [the SCNet guide](docs/elementa_scnet.md) for Windows export transfer,
integrity checks, the existing `$HOME/envs/mlip` environment, and Slurm commands.

From this directory, use:

```bash
python -m cace_sog_training --help
```

The commands are `index`, `environment`, `smoke`, `converge`, `train`, and
`evaluate`. Defaults select `sog_candidates`, `lattice_auxiliary`, and
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
