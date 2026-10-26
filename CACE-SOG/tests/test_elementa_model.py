"""CPU checks for the large-scale adapter; GPU execution is reported separately."""
import io
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
    import torch
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from cace_sog_training.model import atoms_to_data, build_model
    from cace.tools.torch_geometric.batch import Batch
    from cace.data import atomic_data
    RUNTIME_ERROR = None
except ImportError as exc:
    RUNTIME_ERROR = str(exc)


@unittest.skipIf(RUNTIME_ERROR is not None, "training runtime unavailable: " + str(RUNTIME_ERROR))
class ElementaModelTests(unittest.TestCase):
    config = {"n_atom_basis": 2, "n_rbf": 3, "n_radial_basis": 3,
              "max_l": 1, "max_nu": 2, "hidden": [8, 4], "q_hidden": [8, 4],
              "sog_ndl": 2.5, "seed": 11}

    def frame(self, offset=0.0):
        atoms = Atoms("NaCl", positions=[[.2, .3, .4], [1.75 + offset, 1.3, .8]],
                      cell=[[5., 0., 0.], [.4, 5.2, 0.], [.2, .3, 5.4]], pbc=True)
        atoms.calc = SinglePointCalculator(atoms, energy=-5.5 + offset,
                                          forces=np.array([[.01, -.02, .03], [-.01, .02, -.03]]))
        return atoms

    def batch(self):
        return Batch.from_data_list([atoms_to_data(self.frame(value), 3.) for value in (0., .08)])

    def test_model_parameters_are_materialized_before_first_forward(self):
        model = build_model([11, 17], cutoff=3., config=self.config)
        original = [(name, id(value), value.numel()) for name, value in model.named_parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model(self.batch(), training=True)
        after = [(name, id(value), value.numel()) for name, value in model.named_parameters()]
        self.assertEqual(original, after)
        self.assertEqual({id(value) for group in optimizer.param_groups for value in group["params"]},
                         {id(value) for value in model.parameters()})

    def test_combined_force_training_updates_sog_and_short_range_parameters(self):
        model = build_model([11, 17], cutoff=3., config=self.config)
        batch = self.batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        output = model(batch, training=True)
        self.assertEqual(output["energy"].shape, (2,))
        self.assertEqual(output["forces"].shape, (4, 3))
        loss = (output["energy"] - batch.energy).square().mean()
        loss = loss + (output["forces"] - batch.forces).square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        for name in ("sog.wl", "sog.sl"):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(parameter.grad.abs().max().item(), 0, name)
        optimizer.step()
        changed = {name for name, parameter in model.named_parameters()
                   if not torch.equal(before[name], parameter.detach())}
        self.assertTrue(any(name.startswith("short_range.") for name in changed))
        self.assertTrue(any(name.startswith("latent.") for name in changed))
        self.assertIn("sog.wl", changed)
        self.assertIn("sog.sl", changed)

    def test_force_matches_finite_difference_float64_and_preserves_input(self):
        model = build_model([11, 17], cutoff=3., dtype="float64", config=self.config)
        batch = self.batch().to_dict()
        original = batch["positions"].clone()
        output = model(batch, training=False)
        epsilon = 1e-4
        plus, minus = dict(batch), dict(batch)
        plus["positions"] = original.clone().to(torch.float64)
        minus["positions"] = original.clone().to(torch.float64)
        plus["positions"][0, 0] += epsilon
        minus["positions"][0, 0] -= epsilon
        finite_difference = -(model(plus, training=False)["energy"].sum() -
                              model(minus, training=False)["energy"].sum()) / (2 * epsilon)
        self.assertAlmostEqual(output["forces"][0, 0].item(), finite_difference.item(), places=5)
        self.assertTrue(torch.equal(batch["positions"], original))
        self.assertFalse(batch["positions"].requires_grad)
        self.assertFalse(output["energy"].requires_grad)

    def test_state_dict_roundtrip_reproduces_energy_force_and_buffers(self):
        model = build_model([11, 17], cutoff=3., dtype="float64", config=self.config)
        batch = self.batch()
        expected = model(batch, training=False)
        payload = io.BytesIO()
        torch.save({"model_config": model.config, "model_state_dict": model.state_dict()}, payload)
        payload.seek(0)
        try:
            checkpoint = torch.load(payload, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(payload, map_location="cpu")
        saved = checkpoint["model_config"]
        restored = build_model(saved["elements"], saved["cutoff"], saved["model_kind"],
                               dtype=saved["dtype"], config=saved)
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        actual = restored(batch, training=False)
        self.assertTrue(torch.equal(actual["energy"], expected["energy"]))
        self.assertTrue(torch.equal(actual["forces"], expected["forces"]))
        state = restored.state_dict()
        for name in ("sog.amplitude_1", "sog.shift_1", "sog.norm_factor", "sog.wl", "sog.sl"):
            self.assertIn(name, state)
            self.assertEqual(state[name].dtype, torch.float64)

    def test_ase_calculator_labels_are_read_and_global_keys_restored(self):
        atoms = self.frame()
        before = dict(atomic_data.default_data_key)
        converted = atoms_to_data(atoms, cutoff=3., e0={11: -1., 17: -2.})
        self.assertAlmostEqual(converted.energy.item(), -2.5)
        self.assertEqual(atomic_data.default_data_key, before)
        self.assertNotIn("REF_energy", atoms.info)
        self.assertNotIn("REF_forces", atoms.arrays)
        self.assertTrue(np.allclose(converted.forces.numpy(), atoms.calc.results["forces"]))

    def test_requested_float64_conversion_keeps_original_label_precision(self):
        atoms = self.frame()
        atoms.info["REF_energy"] = -5.123456789123
        original_position = float(atoms.positions[0, 0])
        converted = atoms_to_data(atoms, cutoff=3., dtype="float64")
        self.assertEqual(converted.energy.dtype, torch.float64)
        self.assertEqual(converted.forces.dtype, torch.float64)
        self.assertEqual(converted.positions.dtype, torch.float64)
        self.assertEqual(converted.energy.item(), atoms.info["REF_energy"])
        self.assertEqual(converted.positions[0, 0].item(), original_position)

    def test_optional_stress_metadata_does_not_change_ef_batch_keys(self):
        with_stress = self.frame()
        with_stress.info["stress"] = np.eye(3)
        with_stress.info["virials"] = np.eye(3)
        first = atoms_to_data(with_stress, 3.)
        second = atoms_to_data(self.frame(), 3.)
        self.assertIsNone(first.stress)
        self.assertIsNone(first.virials)
        batch = Batch.from_data_list([first, second])
        self.assertEqual(batch.energy.shape, (2,))
        self.assertEqual(batch.forces.shape, (4, 3))
        self.assertIn("stress", with_stress.info)

    def test_adapter_preserves_upstream_periodic_energy_force_and_kernel_gradients(self):
        from cace.modules.sog import SOGPotential
        for self_term in (False, True):
            with self.subTest(remove_self_interaction=self_term):
                original = SOGPotential(N_dl=2.5, Periodic=True, remove_self_interaction=self_term).to("cpu")
                adapted = build_model([11, 17], 3., config=dict(self.config, remove_self_interaction=self_term)).sog
                with torch.no_grad():
                    adapted.wl.copy_(original.wl)
                    adapted.sl.copy_(original.sl)
                cell = torch.tensor(self.frame().cell.array, dtype=torch.float32)
                q1 = torch.tensor([[.3], [-.2]], requires_grad=True)
                q2 = q1.detach().clone().requires_grad_(True)
                r1 = torch.tensor(self.frame().positions, dtype=torch.float32, requires_grad=True)
                r2 = r1.detach().clone().requires_grad_(True)
                e1 = original.compute_potential_SOG_triclinic(r1, q1, cell)
                e2 = adapted.compute_potential_SOG_triclinic(r2, q2, cell)
                f1 = -torch.autograd.grad(e1.sum(), r1, create_graph=True)[0]
                f2 = -torch.autograd.grad(e2.sum(), r2, create_graph=True)[0]
                self.assertTrue(torch.allclose(e1, e2, rtol=2e-5, atol=1e-7))
                self.assertTrue(torch.allclose(f1, f2, rtol=2e-5, atol=1e-7))
                g1 = torch.autograd.grad(e1.sum() + f1.square().sum(), (original.wl, original.sl, q1))
                g2 = torch.autograd.grad(e2.sum() + f2.square().sum(), (adapted.wl, adapted.sl, q2))
                for left, right in zip(g1, g2):
                    self.assertTrue(torch.allclose(left, right, rtol=5e-5, atol=1e-7))

    def test_missing_bad_labels_unknown_baseline_and_boundary_are_rejected(self):
        no_labels = self.frame()
        no_labels.calc = None
        with self.assertRaisesRegex(ValueError, "missing reference"):
            atoms_to_data(no_labels, 3.)
        bad = self.frame()
        bad.info["REF_energy"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            atoms_to_data(bad, 3.)
        with self.assertRaisesRegex(ValueError, "missing train-derived"):
            atoms_to_data(self.frame(), 3., e0={11: 0.})
        slab = self.frame()
        slab.pbc = [True, True, False]
        with self.assertRaisesRegex(ValueError, "3D periodic"):
            atoms_to_data(slab, 3.)

    def test_sr_control_model_and_reciprocal_guard(self):
        model = build_model([11, 17], 3., model_kind="sr", config=self.config)
        output = model(self.batch(), training=True)
        self.assertTrue(torch.equal(output["energy_lr"], torch.zeros(2)))
        self.assertTrue(torch.allclose(output["energy"], output["energy_sr"]))
        guarded = build_model([11, 17], 3., config=dict(self.config, max_reciprocal_modes=1))
        with self.assertRaisesRegex(ValueError, "reciprocal mode guard"):
            guarded(self.batch(), training=False)

    def test_invalid_configuration_and_inference_mode_are_explicit(self):
        for settings in ({"sog_ndl": 0}, {"n_atom_basis": 0}, {"typo": 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                build_model([11, 17], 3., config=settings)
        model = build_model([11, 17], 3., config=self.config)
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, "position gradients"):
            model(self.batch())


if __name__ == "__main__":
    unittest.main()
