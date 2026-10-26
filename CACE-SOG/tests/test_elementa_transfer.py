"""Owned small-data tests for portable index sealing and verified binding."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cace_sog_training import data as data_module
from cace_sog_training.data import DEFAULT_CHANNELS, IndexReader, build_index, _metadata, _put
from cace_sog_training.transfer import seal_index, bind_index, _relative_source
from cace_sog_training import transfer as transfer_module
from test_elementa_data import fixture, frame


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ElementaTransferTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.export = self.base / "Windows 原始 export"
        self.source_index = self.base / "source.sqlite3"
        self.bundle = self.base / "sealed bundle"
        self.target_export = self.base / "SCNet target" / "export"
        self.target_index = self.base / "SCNet target" / "bound.sqlite3"
        fixture(self.export, [
            ("sog_candidates", "train", "train_parent", frame(newline="\r\n")),
            ("sog_candidates", "train", "train_parent", frame(final_newline=False)),
            ("sog_candidates", "val", "val_parent", frame(-9)),
            ("lattice_auxiliary", "test", "test_parent", frame(-8)),
        ])
        build_index(self.export, self.source_index, commit_every=2)

    def seal(self):
        return seal_index(self.source_index, self.bundle, progress=False)

    def move(self):
        shutil.copytree(self.export, self.target_export)
        # Transfer across platforms can change timestamp resolution. Binding
        # must verify content before adopting new local stat values.
        for path in self.target_export.rglob("*"):
            if path.is_file():
                info = path.stat()
                os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 123456789))

    def descriptor(self):
        return json.loads((self.bundle / "bundle.json").read_text())

    def write_descriptor(self, value):
        (self.bundle / "bundle.json").write_text(json.dumps(value), encoding="utf8")

    def test_complete_roundtrip_preserves_offsets_labels_splits_and_original_index(self):
        original = self.source_index.read_bytes()
        sealed = self.seal()
        self.assertEqual(self.source_index.read_bytes(), original)
        self.assertEqual(sealed["indexed_frames"], 4)
        self.assertFalse(sealed["dataset_files_copied"])
        self.assertEqual({path.name for path in self.bundle.iterdir()}, {"index.sqlite3", "bundle.json"})
        with sqlite3.connect(str(self.bundle / "index.sqlite3")) as snapshot:
            metadata = _metadata(snapshot)
        self.assertEqual(metadata["prepared_baseline"]["fit"]["frames"], 2)
        self.assertAlmostEqual(sum(metadata["prepared_baseline"]["e0"].values()), -2.0)
        bundle_bytes = {p.name: p.read_bytes() for p in self.bundle.iterdir()}
        self.move()
        with patch.object(data_module, "_parse_frame", side_effect=AssertionError("bind must not parse frames")):
            bound = bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertEqual(bound["ASE_frames_reparsed"], 0)
        self.assertTrue(bound["prepared_baseline"])
        self.assertEqual(self.source_index.read_bytes(), original)
        self.assertEqual({p.name: p.read_bytes() for p in self.bundle.iterdir()}, bundle_bytes)
        with IndexReader(self.source_index) as source, IndexReader(self.target_index) as target:
            self.assertEqual(target.counts("sog_candidates", "train"), 2)
            self.assertEqual(target.metadata["config"]["export_dir"], str(self.target_export.resolve()))
            self.assertIn("prepared_baseline", target.metadata)
            for split in ("train", "val", "test"):
                first = list(source.iter_records(DEFAULT_CHANNELS, split))
                second = list(target.iter_records(DEFAULT_CHANNELS, split))
                self.assertEqual(len(first), len(second))
                for old, new in zip(first, second):
                    for key in ("channel", "split", "ordinal", "offset", "length", "sha256", "counts_json", "component_id"):
                        self.assertEqual(old[key], new[key])
                    self.assertTrue(Path(new["path"]).is_relative_to(self.target_export.resolve()))
                    self.assertEqual(target.read_atoms(new).info["REF_energy"], new["energy"])

    def test_partial_source_and_existing_bundle_are_refused(self):
        partial = self.base / "partial.sqlite3"
        build_index(self.export, partial, limit_records=1)
        with self.assertRaisesRegex(ValueError, "complete index"):
            seal_index(partial, self.bundle, progress=False)
        self.assertFalse(self.bundle.exists())
        self.seal()
        before = sha(self.bundle / "index.sqlite3")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.seal()
        self.assertEqual(sha(self.bundle / "index.sqlite3"), before)

    def test_tampered_snapshot_refused_before_target_publication(self):
        self.seal()
        self.move()
        path = self.bundle / "index.sqlite3"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_tampered_manifest_refused_before_target_publication(self):
        self.seal()
        self.move()
        path = self.target_export / "manifest.jsonl"
        path.write_bytes(path.read_bytes().replace(b"train_parent", b"other_parent", 1))
        with self.assertRaisesRegex(ValueError, "size mismatch|SHA256 mismatch"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_tampered_extxyz_refused_without_bypassing_hashes(self):
        self.seal()
        self.move()
        path = self.target_export / "sog_candidates/train.extxyz"
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b"1.7", b"1.8", 1))
        # Even preserving size and timestamp cannot bypass content verification.
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_unsafe_relative_paths_missing_file_and_incomplete_marker_refused(self):
        self.seal()
        self.move()
        payload = self.descriptor()
        original = json.loads(json.dumps(payload))
        for value in ("../outside.extxyz", "C:/outside.extxyz", "//server/share/file.extxyz"):
            payload = json.loads(json.dumps(original))
            payload["files"][0]["relative_path"] = value
            self.write_descriptor(payload)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "unsafe"):
                bind_index(self.bundle, self.target_export, self.target_index, progress=False)
            self.assertFalse(self.target_index.exists())
        payload = json.loads(json.dumps(original))
        payload["status"] = "partial"
        self.write_descriptor(payload)
        with self.assertRaisesRegex(ValueError, "complete ELEMENTA"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.write_descriptor(original)
        (self.target_export / "manifest.jsonl").unlink()
        with self.assertRaises(FileNotFoundError):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_duplicate_file_mapping_is_refused(self):
        self.seal()
        self.move()
        payload = self.descriptor()
        payload["files"].append(dict(payload["files"][0]))
        self.write_descriptor(payload)
        with self.assertRaisesRegex(ValueError, "duplicate bundled file_id"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_symlink_escape_is_refused(self):
        self.seal()
        self.move()
        path = self.target_export / "sog_candidates/train.extxyz"
        outside = self.base / "outside.extxyz"
        shutil.copyfile(path, outside)
        path.unlink()
        try:
            path.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not available in this environment")
        with self.assertRaisesRegex(ValueError, "symlink"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())

    def test_existing_target_is_never_overwritten(self):
        self.seal()
        self.move()
        self.target_index.parent.mkdir(exist_ok=True)
        self.target_index.write_bytes(b"owned-existing-index")
        with self.assertRaisesRegex(ValueError, "already exists"):
            bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertEqual(self.target_index.read_bytes(), b"owned-existing-index")

    def test_binding_transaction_failure_leaves_no_published_or_temporary_index(self):
        self.seal()
        self.move()
        original_index = sha(self.source_index)
        original_bundle = {p.name: sha(p) for p in self.bundle.iterdir()}
        put = transfer_module._put
        def fail_config(db, key, value):
            if key == "config":
                raise RuntimeError("owned injected metadata transaction failure")
            return put(db, key, value)
        with patch.object(transfer_module, "_put", side_effect=fail_config):
            with self.assertRaisesRegex(RuntimeError, "metadata transaction failure"):
                bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertFalse(self.target_index.exists())
        self.assertEqual(list(self.target_index.parent.glob(".*.binding-*")), [])
        self.assertEqual(sha(self.source_index), original_index)
        self.assertEqual({p.name: sha(p) for p in self.bundle.iterdir()}, original_bundle)

    def test_windows_source_paths_bind_on_posix_without_editing_records(self):
        self.seal()
        self.move()
        snapshot = self.bundle / "index.sqlite3"
        old_root = str(self.export.resolve())
        windows_root = r"C:\Users\qingu\Documents\elementa_cace_sog_windows\export"
        with sqlite3.connect(str(snapshot)) as db:
            db.row_factory = sqlite3.Row
            metadata = _metadata(db)
            for row in list(db.execute("SELECT id,path FROM files")):
                relative = _relative_source(row["path"], old_root)
                db.execute("UPDATE files SET path=? WHERE id=?", (windows_root + "\\" + relative.replace("/", "\\"), row["id"]))
            config = dict(metadata["config"])
            config.update(export_dir=windows_root, manifest=windows_root + r"\manifest.jsonl")
            _put(db, "config", config)
            _put(db, "export_dir", windows_root)
            db.commit()
        payload = self.descriptor()
        payload["source_export_dir"] = windows_root
        payload["snapshot"]["sha256"] = sha(snapshot)
        payload["snapshot"]["size"] = snapshot.stat().st_size
        self.write_descriptor(payload)
        bound = bind_index(self.bundle, self.target_export, self.target_index, progress=False)
        self.assertEqual(bound["status"], "complete")
        with IndexReader(self.target_index) as reader:
            record = reader.record("sog_candidates", "train", 0)
            self.assertEqual(reader.read_atoms(record).info["REF_energy"], -2.0)
            self.assertEqual(reader.metadata["relocation"]["source_export_dir"], windows_root)


if __name__ == "__main__":
    unittest.main()
