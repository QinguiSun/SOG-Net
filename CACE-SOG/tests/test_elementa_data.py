"""Owned tiny fixtures for disk-backed ELEMENTA training data (no CACE/Torch)."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cace_sog_training.data import build_index, IndexReader, DEFAULT_CHANNELS
from cace_sog_training import data as data_module


def frame(energy=-2.0, forces=True, newline="\n", final_newline=True, pbc="T T T", cell=8,
          metadata=None):
    schema = "species:S:1:pos:R:3" + (":forces:R:3" if forces else "")
    header = 'Lattice="%d 0 0 0 %d 0 0 0 %d" Properties=%s pbc="%s"' % (cell, cell, cell, schema, pbc)
    if energy is not None:
        header += " energy=" + str(energy)
    if metadata:
        header += " " + " ".join(str(key) + "=" + str(value) for key, value in metadata.items())
    lines = ["2", header, "Li 0 0 0" + (" 0.1 0 0" if forces else ""),
             "F 1.7 0 0" + (" -0.1 0 0" if forces else "")]
    return (newline.join(lines) + (newline if final_newline else "")).encode("utf8")


def fixture(root, specs):
    """specs=(channel, split, component, raw) rows; preserve exact frame bytes."""
    root.mkdir(parents=True)
    positions = {}
    records = []
    counts = {}
    for channel, split, component, raw in specs:
        relative = "%s/%s.extxyz" % (channel, split)
        path = root / relative
        path.parent.mkdir(exist_ok=True)
        offset = positions.get(relative, 0)
        with path.open("ab") as stream:
            stream.write(raw)
            if not raw.endswith(b"\n"):
                stream.write(b"\n")
        positions[relative] = path.stat().st_size
        records.append(dict(export_path=relative, export_offset=offset, export_length=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(), natoms=2, channel=channel, split=split,
            component_id=component))
        counts[channel, split] = counts.get((channel, split), 0) + 1
    (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf8")
    summary = {"status": "complete", "counts_by_channel_and_split":
               [dict(channel=channel, split=split, frames=count) for (channel, split), count in counts.items()]}
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf8")
    (root / "run_status.json").write_text('{"status":"complete"}\n', encoding="utf8")
    return records


class ElementaDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.export = self.base / "中文 export with spaces"
        self.database = self.base / "local index" / "index.sqlite3"

    def basic(self):
        return fixture(self.export, [
            ("sog_candidates", "train", "parent_train", frame(newline="\r\n")),
            ("sog_candidates", "train", "parent_train", frame(final_newline=False)),
            ("sog_candidates", "val", "parent_val", frame(-1.9)),
            ("lattice_auxiliary", "test", "parent_test", frame(-1.8)),
            ("spin_review", "train", "parent_spin", frame(-1.7)),
        ])

    def test_exact_offsets_crlf_no_final_newline_unicode_and_ase_calc_labels(self):
        self.basic()
        original = {str(path): path.read_bytes() for path in self.export.rglob("*") if path.is_file()}
        result = build_index(self.export, self.database, commit_every=2)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["indexed_frames"], 4)
        with IndexReader(self.database) as reader:
            self.assertEqual(reader.counts("sog_candidates", "train"), 2)
            self.assertEqual(reader.counts("sog_candidates", "val"), 1)
            self.assertEqual(reader.counts("lattice_auxiliary", "test"), 1)
            self.assertEqual(reader.elements(DEFAULT_CHANNELS, "train"), (3, 9))
            self.assertEqual(reader.metadata["manifest_sha256"], hashlib.sha256((self.export / "manifest.jsonl").read_bytes()).hexdigest())
            records = list(reader.iter_records(DEFAULT_CHANNELS, "train"))
            self.assertEqual(len(records), 2)
            for record in records:
                atoms = reader.read_atoms(record)
                self.assertEqual(atoms.info["REF_energy"], -2.0)
                self.assertTrue(np.allclose(atoms.arrays["REF_forces"], [[.1, 0, 0], [-.1, 0, 0]]))
                self.assertEqual(atoms.get_chemical_symbols(), ["Li", "F"])
                self.assertTrue(atoms.pbc.all())
                self.assertEqual(json.loads(record["counts_json"]), {"3": 1, "9": 1})
            with self.assertRaises(IndexError):
                reader.record("sog_candidates", "train", 2)
            self.assertEqual(reader.record("sog_candidates", "train", 1)["ordinal"], 1)
        self.assertEqual(original, {str(path): path.read_bytes() for path in self.export.rglob("*") if path.is_file()})

    def test_partial_index_is_refused_then_exact_resume_has_no_duplicate_frames(self):
        self.basic()
        partial = build_index(self.export, self.database, commit_every=1, limit_records=2)
        self.assertEqual(partial["status"], "partial")
        with self.assertRaisesRegex(ValueError, "complete index"):
            IndexReader(self.database)
        full = build_index(self.export, self.database, resume=True, commit_every=2)
        self.assertEqual(full["indexed_frames"], 4)
        with sqlite3.connect(str(self.database)) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM records").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT count(DISTINCT manifest_line) FROM records").fetchone()[0], 4)
        repeat = build_index(self.export, self.database, resume=True)
        self.assertEqual(repeat, full)

    def test_missing_energy_or_forces_and_nonfinite_labels_fail(self):
        for label, raw in [("energy", frame(None)), ("forces", frame(forces=False)), ("finite", frame(float("nan")))]:
            with self.subTest(label=label):
                export = self.base / label
                fixture(export, [("sog_candidates", "train", "parent", raw)])
                with self.assertRaisesRegex(ValueError, "missing DFT|nonfinite"):
                    build_index(export, self.base / (label + ".sqlite3"))

    def test_changed_manifest_or_source_is_refused_on_resume(self):
        self.basic()
        build_index(self.export, self.database, limit_records=1, commit_every=1)
        before = self.database.read_bytes()
        manifest = self.export / "manifest.jsonl"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "sources/settings changed"):
            build_index(self.export, self.database, resume=True)
        self.assertEqual(self.database.read_bytes(), before)

    def test_changed_indexed_frame_is_refused_by_reader(self):
        self.basic()
        build_index(self.export, self.database)
        source = self.export / "sog_candidates/train.extxyz"
        source.write_bytes(source.read_bytes().replace(b"1.7", b"1.8"))
        with self.assertRaisesRegex(ValueError, "file changed"):
            IndexReader(self.database)

    def test_exception_rolls_back_only_uncommitted_records_then_resume_recovers(self):
        self.basic()
        parse = data_module._parse_frame
        calls = [0]
        def interrupted(raw, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 3:
                raise RuntimeError("owned injected interruption")
            return parse(raw, *args, **kwargs)
        with patch.object(data_module, "_parse_frame", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                build_index(self.export, self.database, commit_every=2)
        with IndexReader(self.database, require_complete=False) as reader:
            self.assertEqual(reader.counts("sog_candidates", "train"), 2)
            self.assertEqual(reader.counts("sog_candidates", "val"), 0)
            self.assertEqual(reader.metadata["checkpoint"]["line"], 2)
        result = build_index(self.export, self.database, resume=True)
        self.assertEqual(result["indexed_frames"], 4)
        with IndexReader(self.database) as reader:
            self.assertEqual(reader.counts("sog_candidates", "train"), 2)
            self.assertEqual(reader.counts("sog_candidates", "val"), 1)

    def test_changed_source_resume_rejection_preserves_existing_database(self):
        self.basic()
        build_index(self.export, self.database, limit_records=1)
        before = self.database.read_bytes()
        source = self.export / "sog_candidates/train.extxyz"
        source.write_bytes(source.read_bytes().replace(b"1.7", b"1.8"))
        with self.assertRaisesRegex(ValueError, "file changed"):
            build_index(self.export, self.database, resume=True)
        self.assertEqual(self.database.read_bytes(), before)

    def test_export_path_traversal_and_absolute_windows_path_are_refused(self):
        for index, path in enumerate(("../escape.extxyz", "C:\\outside.extxyz", "\\\\server\\share\\file.extxyz")):
            export = self.base / ("unsafe%d" % index)
            records = fixture(export, [("sog_candidates", "train", "parent", frame())])
            records[0]["export_path"] = path
            (export / "manifest.jsonl").write_text(json.dumps(records[0]) + "\n")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "relative"):
                build_index(export, self.base / ("unsafe%d.sqlite3" % index))

    def test_sha_mismatch_and_component_leak_are_refused(self):
        self.basic()
        path = self.export / "manifest.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["sha256"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            build_index(self.export, self.database)
        leak = self.base / "leak"
        fixture(leak, [("sog_candidates", "train", "same", frame()),
                       ("lattice_auxiliary", "test", "same", frame())])
        with self.assertRaisesRegex(ValueError, "crosses train/val/test"):
            build_index(leak, self.base / "leak.sqlite3")

    def test_spin_is_optional_pending_is_not_approved_and_periodicity_is_checked(self):
        self.basic()
        result = build_index(self.export, self.database, channels=("spin_review",))
        self.assertEqual(result["indexed_frames"], 1)
        with IndexReader(self.database) as reader:
            self.assertEqual(reader.counts("spin_review", "train"), 1)
        with self.assertRaisesRegex(ValueError, "unapproved"):
            build_index(self.export, self.base / "pending.sqlite3", channels=("pending_review",))
        slab = self.base / "slab"
        fixture(slab, [("sog_candidates", "train", "parent", frame(pbc="T T F"))])
        with self.assertRaisesRegex(ValueError, "three periodic"):
            build_index(slab, self.base / "slab.sqlite3")

    def test_incomplete_export_and_count_mismatch_are_refused(self):
        self.basic()
        (self.export / "run_status.json").write_text('{"status":"failed"}')
        with self.assertRaisesRegex(ValueError, "not complete"):
            build_index(self.export, self.database)
        (self.export / "run_status.json").write_text('{"status":"complete"}')
        summary = json.loads((self.export / "summary.json").read_text())
        summary["counts_by_channel_and_split"][0]["frames"] += 1
        (self.export / "summary.json").write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "counts differ"):
            build_index(self.export, self.database)
        with self.assertRaisesRegex(ValueError, "complete index"):
            IndexReader(self.database)

    def test_explicit_nonzero_invalid_or_conflicting_charge_is_refused_even_in_spin(self):
        cases = [
            ({"net_charge": 1}, "nonzero physical"),
            ({"total_charge": -2}, "nonzero physical"),
            ({"charge": "nan"}, "nonfinite explicit"),
            ({"net_charge": "not_a_number"}, "invalid explicit"),
            ({"net_charge": 0, "total_charge": 1}, "conflicting explicit"),
        ]
        for number, (metadata, message) in enumerate(cases):
            export = self.base / ("charge%d" % number)
            fixture(export, [("spin_review", "train", "parent", frame(metadata=metadata))])
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, message):
                build_index(export, self.base / ("charge%d.sqlite3" % number), channels=("spin_review",))

    def test_explicit_compatible_units_and_zero_charge_are_kept_missing_charge_not_invented(self):
        fixture(self.export, [
            ("sog_candidates", "train", "zero", frame(metadata={
                "net_charge": 0, "total_charge": "0.0", "charge": "0.0",
                "energy_unit": "eV", "force_unit": "eV/Angstrom", "length_unit": "Angstrom",
                "electronic_converged": "T", "scf_converged": "true"})),
            ("sog_candidates", "train", "unknown", frame()),
        ])
        build_index(self.export, self.database)
        with IndexReader(self.database) as reader:
            self.assertEqual(reader.read_atoms(reader.record("sog_candidates", "train", 0)).info["net_charge"], 0)
            unknown = reader.read_atoms(reader.record("sog_candidates", "train", 1))
            for key in ("net_charge", "total_charge", "charge", "electronic_converged", "scf_converged"):
                self.assertNotIn(key, unknown.info)
            self.assertIn("missing_remains_source_assumption", reader.metadata["config"]["reference_policy"]["charge"])

    def test_incompatible_explicit_units_are_refused(self):
        for number, (key, value) in enumerate((("energy_unit", "Hartree"), ("energy_unit", "eV/atom"),
                                             ("force_unit", "Hartree/Bohr"), ("length_unit", "Bohr"))):
            export = self.base / ("units%d" % number)
            fixture(export, [("sog_candidates", "train", "parent", frame(metadata={key: value}))])
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "incompatible explicit"):
                build_index(export, self.base / ("units%d.sqlite3" % number))

    def test_any_explicit_false_electronic_convergence_is_refused(self):
        for number, metadata in enumerate((
                {"electronic_converged": "false"}, {"scf_converged": "F"},
                {"electronic_converged": 0}, {"electronic_converged": "T", "scf_converged": "false"})):
            export = self.base / ("converged%d" % number)
            fixture(export, [("spin_review", "train", "parent", frame(metadata=metadata))])
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "nonconvergence explicitly"):
                build_index(export, self.base / ("converged%d.sqlite3" % number), channels=("spin_review",))


if __name__ == "__main__":
    unittest.main()
