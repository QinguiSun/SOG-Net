"""Tiny owned export fixtures; never read the user's NAS or SCNet files."""
from collections import Counter
import hashlib
import io
import json

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import numpy as np


def exported_fixture(root, channels=("sog_candidates", "lattice_auxiliary", "pbe_lowdim_auxiliary"),
                     train=5, val=2, test=2, unknown_val=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = []
    for channel_id, channel in enumerate(channels):
        folder = root / channel
        folder.mkdir()
        for split, size in (("train", train), ("val", val), ("test", test)):
            path = folder / (split + ".extxyz")
            with path.open("wb") as stream:
                for index in range(size):
                    distance = 1.8 + .01 * index + .02 * channel_id
                    atoms = Atoms("LiF" if unknown_val and split == "val" else "NaCl", positions=[[0, 0, 0], [distance, .1, .2]],
                                  cell=[6.1, 6.2, 6.3], pbc=True)
                    atoms.calc = SinglePointCalculator(atoms, energy=-3 + .1 * (distance - 1.8)**2,
                        forces=np.array([[.02 * distance, 0, 0], [-.02 * distance, 0, 0]]))
                    buffer = io.StringIO()
                    write(buffer, atoms, format="extxyz")
                    raw = buffer.getvalue().replace("\n", "\r\n").encode("utf8")
                    offset = stream.tell()
                    stream.write(raw)
                    rows.append({"channel": channel, "split": split, "export_path": f"{channel}/{split}.extxyz",
                                 "export_offset": offset, "export_length": len(raw),
                                 "sha256": hashlib.sha256(raw).hexdigest(), "natoms": 2,
                                 "component_id": f"{channel}_{split}_{index}"})
            counts.append({"channel": channel, "split": split, "frames": size})
    with (root / "manifest.jsonl").open("w", encoding="utf8") as manifest:
        for row in rows:
            manifest.write(json.dumps(row) + "\n")
    (root / "summary.json").write_text(json.dumps({"status": "complete",
        "n_exported_frames": len(rows), "counts_by_channel_and_split": counts}), encoding="utf8")
    (root / "run_status.json").write_text('{"status":"complete"}', encoding="utf8")
    return rows
