"""CACE + periodic SOG adapter with explicit, checkpointable parameters.

The upstream scientific modules remain unchanged.  This adapter fixes parameter
registration/device handling at their boundary and computes forces from the
combined energy exactly once.  Latent ``q`` is not a physical spin or charge
label.  Spin data need a consistent electronic-state policy in the data layer.
"""
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch import nn


DEFAULT_CONFIG = {
    "n_atom_basis": 4,
    "n_rbf": 6,
    "n_radial_basis": 6,
    "max_l": 3,
    "max_nu": 3,
    "num_message_passing": 0,
    "hidden": [64, 32],
    "q_hidden": [32, 16],
    "q_channels": 1,
    "sog_ndl": 1.0,
    "remove_self_interaction": False,
    "max_reciprocal_modes": 250000,
    "avg_num_neighbors": 10.0,
    "seed": 20261025,
}


def _dtype(value):
    if value in ("float32", torch.float32):
        return torch.float32
    if value in ("float64", torch.float64):
        return torch.float64
    raise ValueError("dtype must be float32 or float64; AMP is not enabled for force training")


@contextmanager
def _default_dtype(value):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(value)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class _TypedElementEncoder(nn.Module):
    """Use the model dtype even after the process default dtype is restored."""
    def __init__(self, elements):
        super().__init__()
        self.embedding_dim = len(elements)
        mapping = [-1] * (max(elements) + 1)
        for index, element in enumerate(elements):
            mapping[element] = index
        self.register_buffer("index_map", torch.tensor(mapping, dtype=torch.long))
        self.register_buffer("dtype_anchor", torch.empty(0))

    def forward(self, atomic_numbers):
        if atomic_numbers.numel() and (atomic_numbers.min() < 0 or atomic_numbers.max() >= len(self.index_map)):
            raise ValueError("batch contains an element not supported by this model")
        indices = self.index_map[atomic_numbers]
        if (indices < 0).any():
            raise ValueError("batch contains an element absent from the training vocabulary")
        result = torch.zeros((len(indices), self.embedding_dim),
                             dtype=self.dtype_anchor.dtype, device=atomic_numbers.device)
        return result.scatter_(1, indices.reshape(-1, 1), 1)


def _safe_periodic_sog(config):
    # Delayed import: index-only commands do not need the CACE runtime.
    from cace.modules.sog import SOGPotential

    class RegisteredPeriodicSOG(SOGPotential):
        """Preserve the upstream periodic kernel, zero-mode and self-term policy.

        Unlike the original plain tensor attributes, all constants follow
        module.to() and appear in state_dict.  The reciprocal calculation below
        uses input dtype (upstream nvec.float() prevents float64 diagnostics).
        """
        def __init__(self):
            super().__init__(N_dl=config["sog_ndl"], Periodic=True,
                             feature_key="q", output_key="energy_lr",
                             remove_self_interaction=config["remove_self_interaction"],
                             aggregation_mode="sum")
            for name in ("wl", "sl"):
                value = getattr(self, name).detach().clone()
                delattr(self, name)
                self.register_parameter(name, nn.Parameter(value))
            for name in ("amplitude_1", "shift_1", "norm_factor"):
                value = getattr(self, name).detach().clone()
                delattr(self, name)
                self.register_buffer(name, value)
            self.max_reciprocal_modes = int(config["max_reciprocal_modes"])

        def compute_potential_SOG_triclinic(self, positions, q, cell):
            volume = torch.linalg.det(cell)
            if not torch.isfinite(volume) or volume.item() <= 0:
                raise ValueError("periodic SOG requires a finite, right-handed, nondegenerate cell")
            counts = [max(1, int(length.item() / self.N_dl))
                      for length in torch.linalg.vector_norm(cell, dim=1)]
            mode_count = math.prod(2 * count + 1 for count in counts) - 1
            if mode_count > self.max_reciprocal_modes:
                raise ValueError("reciprocal mode guard: %d modes exceed %d; inspect cell/N_dl and "
                                 "choose an explicitly converged budget, do not discard this frame" %
                                 (mode_count, self.max_reciprocal_modes))
            axes = [torch.arange(-count, count + 1, device=positions.device) for count in counts]
            indices = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
            indices = indices[(indices != 0).any(dim=1)]
            first_nonzero = torch.argmax((indices != 0).to(torch.long), dim=1)
            sign = indices.gather(1, first_nonzero[:, None]).squeeze(1)
            indices = indices[sign > 0].to(dtype=positions.dtype)
            reciprocal = (2 * torch.pi) * torch.linalg.inv(cell).T
            k_vectors = indices @ reciprocal
            k_squared = (k_vectors * k_vectors).sum(dim=1)
            phases = positions @ k_vectors.T
            q = q.reshape(len(positions), -1)
            structure_real = q.T @ torch.cos(phases)
            structure_imag = q.T @ torch.sin(phases)
            structure_power = structure_real.square() + structure_imag.square()
            multiplier = (self.wl[None, :] *
                          torch.exp(-k_squared[:, None] * torch.exp(2 * self.sl)[None, :])).sum(dim=1)
            # The nonzero half-space has multiplicity two, as in upstream.
            energy = (2 * multiplier[None, :] * structure_power).sum() / volume
            if self.remove_self_interaction:
                energy = energy - q.square().sum() * multiplier.sum() / (2 * volume)
            return energy.reshape(1) * self.norm_factor

    return RegisteredPeriodicSOG()


def normalize_elements(elements):
    from ase.data import atomic_numbers
    result = []
    for value in elements:
        if isinstance(value, str) and not value.isdigit():
            if value not in atomic_numbers:
                raise ValueError("unknown element: " + value)
            value = atomic_numbers[value]
        if isinstance(value, bool) or int(value) != float(value) or not 1 <= int(value) <= 118:
            raise ValueError("elements must be atomic numbers 1..118 or chemical symbols")
        result.append(int(value))
    result = sorted(set(result))
    if not result:
        raise ValueError("a nonempty training element vocabulary is required")
    return result


class ElementaPotential(nn.Module):
    """A scalar E_SR + E_SOG with one force derivative of their sum."""
    def __init__(self, elements, cutoff, model_kind, config):
        super().__init__()
        from cace.representations import Cace
        from cace.modules import Atomwise, BesselRBF, PolynomialCutoff
        self.config = dict(config, elements=list(elements), cutoff=float(cutoff), model_kind=model_kind)
        encoder = _TypedElementEncoder(elements)
        self.representation = Cace(
            zs=elements, n_atom_basis=config["n_atom_basis"], cutoff=cutoff,
            cutoff_fn=PolynomialCutoff(cutoff=cutoff),
            radial_basis=BesselRBF(cutoff=cutoff, n_rbf=config["n_rbf"], trainable=True),
            n_radial_basis=config["n_radial_basis"], max_l=config["max_l"], max_nu=config["max_nu"],
            num_message_passing=config["num_message_passing"], node_encoder=encoder,
            embed_receiver_nodes=True, type_message_passing=["Bchi"],
            args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
            avg_num_neighbors=config["avg_num_neighbors"], device=torch.device("cpu"),
            atom_embedding_random_seed=[config["seed"], config["seed"] + 1])
        symmetrizer = self.representation.symmetrizer
        n_sym = 1 + sum(len(value) for value in symmetrizer.vec_dict_allnu.values())
        n_in = (config["n_radial_basis"] * n_sym * config["n_atom_basis"] ** 2 *
                (config["num_message_passing"] + 1))
        self.short_range = Atomwise(n_in=n_in, n_layers=len(config["hidden"]) + 1,
                                   n_hidden=config["hidden"], output_key="energy_sr",
                                   add_linear_nn=True, use_batchnorm=False)
        self.latent = None
        self.sog = None
        if model_kind == "sog":
            self.latent = Atomwise(n_in=n_in, n_out=config["q_channels"],
                                   n_layers=len(config["q_hidden"]) + 1, n_hidden=config["q_hidden"],
                                   per_atom_output_key="q", output_key="total_q", bias=False,
                                   add_linear_nn=True, use_batchnorm=False)
            self.sog = _safe_periodic_sog(config)

    def forward(self, data, training=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("forces require position gradients; use model.eval(), not inference_mode()")
        training = self.training if training is None else bool(training)
        if hasattr(data, "to_dict"):
            data = data.to_dict()
        local = dict(data)
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        for key, value in tuple(local.items()):
            if isinstance(value, torch.Tensor):
                local[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        with torch.enable_grad():
            positions = local["positions"].detach().clone().requires_grad_(True)
            local["positions"] = positions
            if local.get("batch") is None:
                local["batch"] = torch.zeros(len(positions), dtype=torch.long, device=device)
            self.representation.device = device
            output = self.short_range(self.representation(local))
            sr = output["energy_sr"].reshape(-1)
            if self.sog is not None:
                output = self.sog(self.latent(output))
                lr = output["energy_lr"].reshape(-1)
            else:
                lr = torch.zeros_like(sr)
            energy = sr + lr
            derivative = torch.autograd.grad(energy.sum(), positions, create_graph=training,
                                             retain_graph=training, allow_unused=True)[0]
            forces = -derivative if derivative is not None else torch.zeros_like(positions)
        result = {"energy": energy, "forces": forces, "energy_sr": sr, "energy_lr": lr}
        return result if training else {key: value.detach() for key, value in result.items()}


def build_model(elements, cutoff=5.0, model_kind="sog", device="cpu", dtype="float32", config=None):
    """Create a fully materialized model before constructing an optimizer.

    Pass ``model.config`` back as config to reconstruct a saved state_dict. The
    explicit outer arguments take precedence over these identifying fields.
    """
    elements = normalize_elements(elements)
    if model_kind not in ("sog", "sr") or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("model_kind must be sog/sr and cutoff must be finite and positive")
    requested = dict(config or {})
    for key in ("elements", "cutoff", "model_kind", "dtype"):
        requested.pop(key, None)
    unknown = set(requested) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError("unknown model configuration keys: " + ", ".join(sorted(unknown)))
    settings = dict(DEFAULT_CONFIG, **requested)
    for key in ("n_atom_basis", "n_rbf", "n_radial_basis", "q_channels", "max_reciprocal_modes"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(key + " must be a positive integer")
    if not 0 <= settings["max_l"] <= 6 or not 1 <= settings["max_nu"] <= 4:
        raise ValueError("max_l must be 0..6 and max_nu 1..4")
    if not isinstance(settings["num_message_passing"], int) or settings["num_message_passing"] < 0:
        raise ValueError("num_message_passing must be a nonnegative integer")
    for key in ("sog_ndl", "avg_num_neighbors"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(key + " must be finite and positive")
    for key in ("hidden", "q_hidden"):
        settings[key] = list(settings[key])
        if not settings[key] or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in settings[key]):
            raise ValueError(key + " must contain positive layer widths")
    target_dtype = _dtype(dtype)
    # Upstream embedding constructors reseed torch; keep caller RNG state intact.
    chosen_device = torch.device(device)
    devices = ([chosen_device.index if chosen_device.index is not None else torch.cuda.current_device()]
               if chosen_device.type == "cuda" else [])
    with torch.random.fork_rng(devices=devices), _default_dtype(target_dtype):
        torch.manual_seed(int(settings["seed"]))
        model = ElementaPotential(elements, float(cutoff), model_kind, settings)
        model = model.to(device=device, dtype=target_dtype)
    model.config["dtype"] = str(target_dtype).split(".")[-1]
    return model


def atoms_to_data(atoms, cutoff, e0=None, dtype=None):
    """Validate labels and convert one frame without mutating global data keys."""
    from cace.data import AtomicData
    from cace.data import atomic_data as atomic_module
    energy = atoms.info.get("REF_energy", atoms.info.get("energy"))
    forces = atoms.arrays.get("REF_forces", atoms.arrays.get("forces"))
    calculator = getattr(atoms, "calc", None)
    if calculator is not None:
        if energy is None:
            energy = calculator.results.get("energy")
        if forces is None:
            forces = calculator.results.get("forces")
    if energy is None or forces is None:
        raise ValueError("frame is missing reference energy or forces")
    energy = np.asarray(energy)
    forces = np.asarray(forces, dtype=float)
    if energy.size != 1 or not np.isfinite(energy).all() or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError("reference energy/force labels are nonfinite or have invalid shapes")
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
        raise ValueError("nonfinite geometry")
    if not atoms.pbc.all() or np.linalg.det(atoms.cell.array) <= 1e-8:
        raise ValueError("the periodic training adapter requires right-handed 3D periodic cells")
    baseline = None
    if e0 is not None:
        baseline = {int(key): float(value) for key, value in e0.items()}
        if any(not math.isfinite(value) for value in baseline.values()):
            raise ValueError("nonfinite elemental energy baseline")
        missing = set(map(int, atoms.numbers)) - set(baseline)
        if missing:
            raise ValueError("missing train-derived elemental baseline for Z=" + str(sorted(missing)))
    frame = atoms.copy()
    frame.info["REF_energy"] = float(energy.reshape(-1)[0])
    frame.set_array("REF_forces", forces.copy())
    saved_keys = dict(atomic_module.default_data_key)
    try:
        # This adapter supervises E/F only. Optional per-frame stress/virial or
        # molecular fields must not make graph keys depend on the first item
        # of a batch. They remain intact in the source extxyz and index.
        keys = {key: None for key in saved_keys}
        keys.update(energy="REF_energy", forces="REF_forces")
        with _default_dtype(_dtype(dtype) if dtype is not None else torch.get_default_dtype()):
            converted = AtomicData.from_atoms(frame, cutoff=cutoff,
                                             data_key=keys,
                                             atomic_energies=baseline)
    finally:
        atomic_module.default_data_key.clear()
        atomic_module.default_data_key.update(saved_keys)
    if converted.energy is None or converted.forces is None:
        raise ValueError("AtomicData conversion lost reference labels")
    return converted
