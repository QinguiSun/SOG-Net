"""Single-device training, exact restart positions, and streaming metrics.

The legacy examples remain available; this module deliberately does not use
their eager loaders, epoch-sized prediction lists, or model-saving helpers.
"""
from bisect import bisect_right
from collections import deque
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data import DEFAULT_CHANNELS, IndexReader
from .model import atoms_to_data, build_model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def environment():
    versions = {}
    for name in ("torch", "numpy", "ase", "matscipy", "pytorch-finufft", "lightning"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"],
                                            cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": versions, "git_commit": revision, "git_dirty": dirty,
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def peak_memory(device):
    # macOS reports bytes, Linux reports KiB. SCNet is Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if platform.system() == "Darwin" else rss * 1024)
    return {"process_peak_rss_bytes": rss_bytes,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == "cuda" else 0}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_scope(channels, spin_policy="exclude", spin_policy_note=None):
    channels = tuple(channels)
    allowed = set(DEFAULT_CHANNELS) | {"spin_review"}
    if not channels or len(set(channels)) != len(channels) or set(channels) - allowed:
        raise ValueError("choose distinct approved channels; pending/charge/boundary review channels are not training inputs")
    if "spin_review" in channels and (spin_policy != "consistent-state" or not spin_policy_note):
        raise ValueError("spin_review requires --spin-policy consistent-state and --spin-policy-note describing the reviewed state policy; this does not add spin inputs")
    return channels


def index_identity(reader):
    encoded = json.dumps(reader.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf8")).hexdigest()


class FrameDataset(Dataset):
    """Only channel counts are resident. SQLite maps each ordinal to raw bytes."""
    def __init__(self, database, channels, split, cutoff, e0, dtype="float32"):
        self.database = str(Path(database).resolve())
        self.channels = tuple(channels)
        self.split = split
        self.cutoff = cutoff
        self.dtype = dtype
        self.e0 = {int(k): float(v) for k, v in e0.items()}
        self.ends = []
        total = 0
        reader = IndexReader(self.database)
        try:
            for channel in channels:
                total += reader.counts(channel, split)
                self.ends.append(total)
        finally:
            reader.close()
        self.total = total
        self._reader = None
        self._pid = None

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if not 0 <= index < self.total:
            raise IndexError(index)
        if self._reader is None or self._pid != os.getpid():
            if self._reader is not None:
                self._reader.close()
            self._reader = IndexReader(self.database)
            self._pid = os.getpid()
        group = bisect_right(self.ends, index)
        start = self.ends[group - 1] if group else 0
        record = self._reader.record(self.channels[group], self.split, index - start)
        atoms = self._reader.read_atoms(record)
        missing = set(int(z) for z in atoms.numbers) - self.e0.keys()
        if missing:
            raise ValueError("elements absent from selected train: " + str(sorted(missing)))
        graph = atoms_to_data(atoms, cutoff=self.cutoff, e0=self.e0, dtype=self.dtype)
        graph.channel_id = torch.tensor([group], dtype=torch.long)
        return graph

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reader"] = None
        state["_pid"] = None
        return state

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None


class EpochSampler(Sampler):
    """An O(1)-memory affine permutation; every sample occurs exactly once.

    It is a deterministic permutation, not a uniformly sampled permutation of
    all N! possibilities. The next ordinal is sufficient for an exact restart.
    """
    def __init__(self, size, seed, epoch, start=0, shuffle=True):
        if not 0 <= start <= size:
            raise ValueError("invalid sampler restart position")
        self.size, self.start, self.shuffle = size, start, shuffle
        rng = random.Random(seed + 1000003 * epoch)
        self.a, self.b = 1, 0
        if shuffle and size > 1:
            self.a = rng.randrange(1, size)
            while math.gcd(self.a, size) != 1:
                self.a = rng.randrange(1, size)
            self.b = rng.randrange(size)

    def __len__(self):
        return self.size - self.start

    def __iter__(self):
        for ordinal in range(self.start, self.size):
            yield (self.a * ordinal + self.b) % self.size if self.shuffle else ordinal


def loader_for(dataset, batch_size, workers=0, sampler=None, seed=0):
    from cace.tools.torch_geometric.dataloader import DataLoader
    kwargs = {"dataset": dataset, "batch_size": batch_size, "num_workers": workers,
              "shuffle": False, "drop_last": False,
              "generator": torch.Generator().manual_seed(seed)}
    if sampler is not None:
        kwargs["sampler"] = sampler
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=False,
                      multiprocessing_context="spawn")
    return DataLoader(**kwargs)


def train_baseline(reader, channels, progress_seconds=30):
    """Reuse the sealed Windows fit, or stream train labels for a new scope."""
    from .baseline import fit_baseline, prepared_baseline
    prepared = prepared_baseline(reader, channels)
    if prepared is not None:
        print("Using sealed train-only energy baseline", flush=True)
        return prepared
    return fit_baseline(reader, channels, progress_seconds)


class StreamingMetrics:
    def __init__(self, state=None):
        self.state = dict(state or {})

    def update(self, name, energy_errors, force_errors):
        e = energy_errors.detach().double()
        f = force_errors.detach().double()
        if not torch.isfinite(e).all() or not torch.isfinite(f).all():
            raise ValueError("nonfinite prediction/error")
        row = self.state.setdefault(name, {"frames": 0, "force_components": 0,
                                          "e_abs": 0., "e_sq": 0., "f_abs": 0., "f_sq": 0.})
        row["frames"] += e.numel()
        row["force_components"] += f.numel()
        row["e_abs"] += e.abs().sum().item()
        row["e_sq"] += e.square().sum().item()
        row["f_abs"] += f.abs().sum().item()
        row["f_sq"] += f.square().sum().item()

    def result(self):
        return {name: {"frames": row["frames"], "atoms": row["force_components"] // 3,
                       "energy_MAE_eV_atom": row["e_abs"] / row["frames"],
                       "energy_RMSE_eV_atom": math.sqrt(row["e_sq"] / row["frames"]),
                       "force_component_MAE_eV_A": row["f_abs"] / row["force_components"],
                       "force_component_RMSE_eV_A": math.sqrt(row["f_sq"] / row["force_components"])}
                for name, row in self.state.items() if row["frames"] and row["force_components"]}


def batch_errors(batch, prediction):
    counts = (batch.ptr[1:] - batch.ptr[:-1]).to(prediction["energy"].dtype)
    e = (prediction["energy"].reshape(-1) - batch.energy.reshape(-1)) / counts
    f = prediction["forces"] - batch.forces
    return e, f


def record_metrics(metrics, batch, channels, e, f):
    metrics.update("all", e, f)
    channel_ids = batch.channel_id.reshape(-1)
    for channel_id in torch.unique(channel_ids).tolist():
        graph_mask = channel_ids == channel_id
        atom_mask = graph_mask[batch.batch]
        metrics.update(channels[channel_id], e[graph_mask], f[atom_mask])


def evaluate_loader(model, loader, channels, device, progress_seconds=30):
    model.eval()
    metrics = StreamingMetrics()
    last = time.monotonic()
    # E/F evaluation needs position autograd; never use inference_mode here.
    with torch.enable_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction = model(batch.to_dict(), training=False)
            e, f = batch_errors(batch, prediction)
            record_metrics(metrics, batch, channels, e, f)
            if time.monotonic() - last >= progress_seconds:
                print(f"evaluation: {metrics.state['all']['frames']:,} frames", flush=True)
                last = time.monotonic()
            del prediction, e, f, batch
    return metrics.result()


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_checkpoint(path):
    # Only load checkpoints created by this training command in a trusted run.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "elementa-cace-sog-v1":
        raise ValueError("not an ELEMENTA training checkpoint")
    return checkpoint


def score(metrics, energy_weight, force_weight):
    values = metrics["all"]
    return (energy_weight * values["energy_RMSE_eV_atom"]**2 +
            force_weight * values["force_component_RMSE_eV_A"]**2)


@contextmanager
def run_lock(folder):
    """POSIX advisory lock: a killed Slurm job never leaves a stale lock."""
    import fcntl
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".train.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another training process is using this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_train(args):
    with run_lock(args.run_dir):
        return _run_train(args)


def _run_train(args):
    channels = validate_scope(args.channels, args.spin_policy, args.spin_policy_note)
    run = Path(args.run_dir).resolve()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this release is single-device; do not launch multiple torchrun ranks against one run")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.checkpoint_every < 1:
        raise ValueError("invalid epochs/batch/workers/checkpoint settings")
    if args.energy_weight < 0 or args.force_weight < 0 or args.energy_weight + args.force_weight <= 0:
        raise ValueError("loss weights must be nonnegative with at least one positive")
    if args.lr <= 0 or args.max_grad_norm <= 0:
        raise ValueError("learning rate and gradient clip must be positive")
    if args.validate_every_steps < 0 or args.seed < 0:
        raise ValueError("validation interval and seed must be nonnegative")
    if args.resume:
        saved = load_checkpoint(args.resume)
    else:
        saved = None
        if run.exists() and any(path.name != ".train.lock" for path in run.iterdir()):
            raise ValueError("run directory must be new/empty unless --resume is provided")
    run.mkdir(parents=True, exist_ok=True)
    reader = IndexReader(args.database)
    try:
        identity = index_identity(reader)
        scope = {"channels": list(channels), "spin_policy": args.spin_policy,
                 "spin_policy_note": args.spin_policy_note, "index_identity": identity,
                 "batch_size": args.batch_size, "seed": args.seed, "shuffle": args.shuffle,
                 "energy_weight": args.energy_weight, "force_weight": args.force_weight,
                 "max_grad_norm": args.max_grad_norm, "lr": args.lr}
        if saved:
            if saved["scope"] != scope:
                raise ValueError("resume data/split/batch/seed/loss/optimizer settings differ; reuse the original arguments")
            e0, baseline = saved["e0"], saved["baseline"]
            config = saved["model_config"]
            if args.cutoff != config["cutoff"] or args.dtype != config["dtype"] or args.model_kind != config["model_kind"]:
                raise ValueError("resume model settings differ from checkpoint")
            requested_model = {"sog_ndl": args.n_dl, "max_reciprocal_modes": args.max_modes,
                               "n_atom_basis": args.atom_basis, "n_rbf": args.radial_basis,
                               "n_radial_basis": args.radial_basis, "max_l": args.max_l,
                               "max_nu": args.max_nu, "num_message_passing": args.message_passing,
                               "seed": args.seed}
            if any(config[key] != value for key, value in requested_model.items()):
                raise ValueError("resume architecture/grid settings differ; reuse the original arguments")
        else:
            seed_everything(args.seed)
            e0, baseline = train_baseline(reader, channels, args.progress_seconds)
            config = {"sog_ndl": args.n_dl, "max_reciprocal_modes": args.max_modes,
                      "n_atom_basis": args.atom_basis, "n_rbf": args.radial_basis, "n_radial_basis": args.radial_basis,
                      "max_l": args.max_l, "max_nu": args.max_nu,
                      "num_message_passing": args.message_passing, "seed": args.seed}
        for channel in channels:
            if reader.counts(channel, "train") == 0:
                raise ValueError(f"selected channel has no train frames: {channel}")
        # The dataset also checks support at each conversion. Validate index
        # element inventories up front when available, before a long epoch.
        if hasattr(reader, "elements"):
            supported = {int(z) for z in e0}
            for split in ("val", "test"):
                unseen = set(reader.elements(channels, split)) - supported
                if unseen:
                    raise ValueError(f"{split} has elements absent from selected train: {sorted(unseen)}")
        model = build_model(sorted(int(z) for z in e0), cutoff=args.cutoff,
                            model_kind=args.model_kind, device=args.device,
                            dtype=args.dtype, config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        epoch, offset, step, best = 0, 0, 0, None
        train_metrics = StreamingMetrics()
        if saved:
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            epoch, offset, step, best = (saved[k] for k in ("epoch", "offset", "step", "best_val_score"))
            if args.epochs < epoch:
                raise ValueError("requested epochs precede the saved training progress")
            train_metrics = StreamingMetrics(saved.get("train_metrics"))
            restore_rng(saved["rng"])
        info = {"scope": scope, "model_config": model.config, "baseline": baseline,
                "e0": e0, "environment": environment(), "index_metadata": reader.metadata,
                "epochs_requested": args.epochs, "all_selected_train_frames_per_epoch": True,
                "spin_policy_is_user_review_declaration_not_explicit_spin_input": True}
        write_json(run / "run_config.json", info)
    finally:
        reader.close()
    train_set = FrameDataset(args.database, channels, "train", args.cutoff, e0, args.dtype)
    val_set = FrameDataset(args.database, channels, "val", args.cutoff, e0, args.dtype)
    if not len(val_set):
        raise ValueError("selected val is empty; cannot select a validation-best checkpoint")
    val_loader = loader_for(val_set, args.batch_size, args.workers, seed=args.seed)
    wall_start, last = time.monotonic(), time.monotonic()

    def state():
        return {"format": "elementa-cace-sog-v1", "model_config": model.config,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "e0": e0, "baseline": baseline, "scope": scope,
                "epoch": epoch, "offset": offset, "step": step, "best_val_score": best,
                "train_metrics": train_metrics.state, "rng": rng_state(), "environment": environment()}

    def validate_and_save():
        nonlocal best
        metrics = evaluate_loader(model, val_loader, channels, args.device, args.progress_seconds)
        value = score(metrics, args.energy_weight, args.force_weight)
        if not math.isfinite(value):
            raise ValueError("nonfinite validation score")
        if best is None or value < best:
            best = value
            atomic_checkpoint(run / "best.pt", state())
        write_json(run / "val_latest.json", {"epoch": epoch, "step": step, "score": value,
                                            "metrics": metrics, "best_val_score": best})
        model.train()
        return metrics

    write_json(run / "run_status.json", {"status": "running", "epoch": epoch, "offset": offset, "step": step})
    try:
        if args.max_steps is not None and step >= args.max_steps:
            result = {"status": "partial", "reason": "checkpoint already reached explicit max_steps", "step": step,
                      "epoch": epoch, "offset": offset, **peak_memory(args.device)}
            write_json(run / "run_status.json", result)
            return result
        while epoch < args.epochs:
            sampler = EpochSampler(len(train_set), args.seed, epoch, offset, args.shuffle == "affine")
            loader = loader_for(train_set, args.batch_size, args.workers, sampler, args.seed + epoch)
            model.train()
            for batch in loader:
                batch = batch.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch.to_dict(), training=True)
                e, f = batch_errors(batch, prediction)
                loss = args.energy_weight * e.square().mean() + args.force_weight * f.square().mean()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss; no frames were silently skipped")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                record_metrics(train_metrics, batch, channels, e, f)
                offset += int(batch.energy.numel())
                step += 1
                loss_value = float(loss.detach())
                del prediction, e, f, loss, batch
                if step == 1 or step % args.checkpoint_every == 0:
                    atomic_checkpoint(run / "last.pt", state())
                if args.validate_every_steps and step % args.validate_every_steps == 0:
                    validate_and_save()
                if time.monotonic() - last >= args.progress_seconds:
                    progress = {"status": "running", "epoch": epoch, "offset": offset,
                                "train_frames_per_epoch": len(train_set), "step": step,
                                "loss": loss_value, "gradient_norm": float(norm), **peak_memory(args.device)}
                    write_json(run / "run_status.json", progress)
                    print(json.dumps(progress), flush=True)
                    last = time.monotonic()
                if args.max_steps is not None and step >= args.max_steps:
                    atomic_checkpoint(run / "last.pt", state())
                    result = {"status": "partial", "reason": "explicit max_steps debug stop", "step": step,
                              "epoch": epoch, "offset": offset, **peak_memory(args.device)}
                    write_json(run / "run_status.json", result)
                    return result
            if offset != len(train_set):
                raise ValueError("epoch coverage mismatch")
            metrics = validate_and_save()
            with (run / "epochs.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps({"epoch": epoch, "step": step, "train": train_metrics.result(),
                                         "val": metrics}, allow_nan=False) + "\n")
            epoch += 1
            offset = 0
            train_metrics = StreamingMetrics()
            atomic_checkpoint(run / "last.pt", state())
        result = {"status": "complete", "epochs": epoch, "step": step, "best_val_score": best,
                  "elapsed_seconds_this_invocation": time.monotonic() - wall_start,
                  "train_frames_per_epoch": len(train_set), **peak_memory(args.device)}
        write_json(run / "run_status.json", result)
        return result
    except BaseException as exc:
        # Keep the last fully completed optimizer-step checkpoint. A failed
        # optimizer step is never advertised as a safe restart position.
        write_json(run / "run_status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                             "error": str(exc), "last_checkpoint": str(run / "last.pt")})
        raise
    finally:
        train_set.close()
        val_set.close()


def run_evaluate(args):
    saved = load_checkpoint(args.checkpoint)
    channels = tuple(args.channels or saved["scope"]["channels"])
    validate_scope(channels, saved["scope"]["spin_policy"], saved["scope"]["spin_policy_note"])
    reader = IndexReader(args.database)
    try:
        if index_identity(reader) != saved["scope"]["index_identity"]:
            raise ValueError("evaluation index differs from training checkpoint")
    finally:
        reader.close()
    config = saved["model_config"]
    model = build_model(config["elements"], cutoff=config["cutoff"], model_kind=config["model_kind"],
                        device=args.device, dtype=config["dtype"], config=config)
    model.load_state_dict(saved["model_state"], strict=True)
    dataset = FrameDataset(args.database, channels, args.split, config["cutoff"], saved["e0"], config["dtype"])
    if not len(dataset):
        raise ValueError("evaluation split is empty")
    try:
        result = {"status": "complete", "split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
                  "metrics": evaluate_loader(model, loader_for(dataset, args.batch_size, args.workers),
                                             channels, args.device, args.progress_seconds), **peak_memory(args.device)}
    finally:
        dataset.close()
    destination = Path(args.output)
    write_json(destination / "metrics.json", result)
    return result


def run_smoke(args):
    """Actual total-force backward, parameter updates, and strict state reload."""
    channels = validate_scope(args.channels, args.spin_policy, args.spin_policy_note)
    seed_everything(args.seed)
    reader = IndexReader(args.database)
    try:
        # Smoke intentionally reads only a small prefix; formal train never caps.
        elements = set()
        picked = []
        for record in reader.iter_records(channels, "train"):
            picked.append(record)
            elements.update(int(z) for z in json.loads(record["counts_json"]))
            if len(picked) >= args.batch_size:
                break
        if not picked:
            raise ValueError("smoke train is empty")
        e0 = {z: 0. for z in sorted(elements)}
        from cace.tools.torch_geometric.batch import Batch
        batch = Batch.from_data_list([atoms_to_data(reader.read_atoms(row), args.cutoff, e0, dtype=args.dtype) for row in picked]).to(args.device)
    finally:
        reader.close()
    config = {"sog_ndl": args.n_dl, "max_reciprocal_modes": args.max_modes, "n_atom_basis": args.atom_basis,
              "n_rbf": args.radial_basis, "n_radial_basis": args.radial_basis, "max_l": args.max_l, "max_nu": args.max_nu,
              "num_message_passing": args.message_passing, "seed": args.seed}
    model = build_model(sorted(elements), cutoff=args.cutoff, device=args.device, dtype=args.dtype, config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tracked = {name: value.detach().clone() for name, value in model.named_parameters()}
    memory_trace = deque(maxlen=32)
    steps = getattr(args, "steps", 10)
    if not 1 <= steps <= 100000:
        raise ValueError("smoke steps must be between 1 and 100000")
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch.to_dict(), training=True)
        e, f = batch_errors(batch, prediction)
        loss = e.square().mean() + f.square().mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite smoke training loss")
        loss.backward()
        gradients = {name: float(value.grad.detach().norm()) for name, value in model.named_parameters()
                     if value.grad is not None}
        if not gradients or not all(math.isfinite(value) for value in gradients.values()):
            raise ValueError("missing/nonfinite parameter gradients")
        loss_value = float(loss.detach())
        if step == 0:
            initial_loss = loss_value
        optimizer.step()
        del prediction, e, f, loss
        memory_trace.append({"step": step + 1, **peak_memory(args.device),
                             "cuda_allocated_bytes": torch.cuda.memory_allocated(args.device)
                             if torch.device(args.device).type == "cuda" else 0})
    changed = [name for name, value in model.named_parameters() if not torch.equal(value.detach(), tracked[name])]
    kernel = {name: value for name, value in gradients.items() if name.endswith(("wl", "sl"))}
    if not changed or not kernel or not any(value > 0 for value in kernel.values()):
        raise ValueError("optimizer or SOG kernel did not receive an update")
    reference = model(batch.to_dict(), training=False)
    output = Path(args.output)
    checkpoint = output / "smoke.pt"
    atomic_checkpoint(checkpoint, {"model_config": model.config, "model_state": model.state_dict()})
    other = build_model(sorted(elements), cutoff=args.cutoff, device=args.device, dtype=args.dtype, config=model.config)
    other.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=True)["model_state"], strict=True)
    replay = other(batch.to_dict(), training=False)
    errors = {key: float((reference[key] - replay[key]).detach().abs().max()) for key in ("energy", "forces")}
    tolerance = 1e-5 if args.dtype == "float32" else 1e-9
    if max(errors.values()) > tolerance:
        raise ValueError("checkpoint energy/force reload differs")
    result = {"status": "complete", "frames": len(picked), "updated_parameter_count": len(changed),
              "kernel_gradient_norms": kernel, "reload_max_abs_errors": errors,
              "optimizer_steps": steps, "loss_before_first_update": initial_loss,
              "loss_before_last_update": loss_value, "memory_trace_last_32_steps": list(memory_trace),
              **peak_memory(args.device), "environment": environment()}
    write_json(output / "smoke.json", result)
    return result


def run_converge(args):
    """Fixed-parameter quadrature comparison, never train one model per grid."""
    if len(args.n_dl) < 2 or min(args.n_dl) <= 0 or args.samples < 1:
        raise ValueError("provide at least two positive N_dl values and positive sample count")
    channels = validate_scope(args.channels, args.spin_policy, args.spin_policy_note)
    reader = IndexReader(args.database)
    atoms = []
    try:
        elements = set()
        for record in reader.iter_records(channels, "train"):
            atoms.append(reader.read_atoms(record))
            elements.update(int(z) for z in json.loads(record["counts_json"]))
            if len(atoms) >= args.samples:
                break
    finally:
        reader.close()
    if not atoms:
        raise ValueError("convergence train is empty")
    from cace.tools.torch_geometric.batch import Batch
    seed_everything(args.seed)
    checkpoint = load_checkpoint(args.checkpoint) if args.checkpoint else None
    if checkpoint:
        reader = IndexReader(args.database)
        try:
            if index_identity(reader) != checkpoint["scope"]["index_identity"]:
                raise ValueError("convergence index differs from checkpoint")
        finally:
            reader.close()
        config = checkpoint["model_config"]
        if config["model_kind"] != "sog":
            raise ValueError("grid convergence requires a SOG model")
        model = build_model(config["elements"], cutoff=config["cutoff"], device=args.device,
                            dtype=config["dtype"], config=config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        e0 = checkpoint["e0"]
        cutoff, dtype = config["cutoff"], config["dtype"]
    else:
        model = build_model(sorted(elements), cutoff=args.cutoff, device=args.device, dtype=args.dtype,
                            config={"sog_ndl": max(args.n_dl), "max_reciprocal_modes": args.max_modes,
                                    "n_atom_basis": args.atom_basis, "n_rbf": args.radial_basis, "n_radial_basis": args.radial_basis,
                                    "max_l": args.max_l, "max_nu": args.max_nu,
                                    "num_message_passing": args.message_passing, "seed": args.seed})
        e0 = {z: 0. for z in elements}
        cutoff, dtype = args.cutoff, args.dtype
    outputs = {}
    for n_dl in sorted(set(args.n_dl), reverse=True):
        model.sog.N_dl = n_dl
        values = []
        started = time.monotonic()
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(args.device)
        for item in atoms:
            batch = Batch.from_data_list([atoms_to_data(item, cutoff, e0, dtype=dtype)]).to(args.device)
            prediction = model(batch.to_dict(), training=False)
            values.append((prediction["energy"].detach().cpu().numpy() / len(item),
                           prediction["forces"].detach().cpu().numpy()))
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        outputs[n_dl] = (values, time.monotonic() - started, peak_memory(args.device))
    finest = min(outputs)
    reference = outputs[finest][0]
    rows = []
    for n_dl, (values, seconds, memory) in outputs.items():
        e_error = max(float(np.max(np.abs(a[0] - b[0]))) for a, b in zip(values, reference))
        f_error = max(float(np.max(np.abs(a[1] - b[1]))) for a, b in zip(values, reference))
        rows.append({"n_dl": n_dl, "energy_max_abs_eV_atom_vs_finest": e_error,
                     "force_max_abs_eV_A_vs_finest": f_error, "elapsed_seconds": seconds, **memory})
    result = {"status": "complete", "samples": len(atoms), "finest_n_dl": finest, "grids": rows,
              "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "interpretation": "fixed-parameter quadrature comparison against finest supplied grid; not a proof of convergence beyond tested grids/cells"}
    write_json(Path(args.output) / "convergence.json", result)
    return result
