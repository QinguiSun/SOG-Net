"""Portable commands work in the Windows CPU environment without the ML stack."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from elementa_fixtures import exported_fixture


ROOT = Path(__file__).resolve().parents[1]


def cpu_command(*args):
    # resource is unavailable on Windows; forbid loading the Linux training
    # runtime even on CI runners that happen to have PyTorch installed.
    code = """
import importlib.abc, sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','cace','resource'):
            raise RuntimeError('CPU preprocessing imported '+fullname)
sys.meta_path.insert(0, NoML())
from cace_sog_training.__main__ import main
sys.exit(main(sys.argv[1:]))
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code, *map(str, args)],
                            cwd=ROOT, env=env, text=True, encoding="utf-8", capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cpu_only_index_seal_bind_and_preserved_frame_read(tmp_path):
    export = tmp_path / "Windows 原始 export"
    exported_fixture(export, train=1, val=1, test=1)
    source = tmp_path / "training.sqlite3"
    cpu_command("index", "--export-dir", export, "--database", source)
    bundle = tmp_path / "bundle"
    cpu_command("seal", "--database", source, "--bundle", bundle)
    remote = tmp_path / "SCNet copy" / "export"
    shutil.copytree(export, remote)
    # The target mtime and absolute root differ. Full SHA checks, followed by
    # atomic rebinding, must still accept these unchanged bytes.
    for path in remote.rglob("*"):
        if path.is_file():
            os.utime(path, (1700000000, 1700000000))
    destination = tmp_path / "bound.sqlite3"
    cpu_command("bind", "--bundle", bundle, "--export-dir", remote, "--database", destination)
    from cace_sog_training.data import IndexReader
    with IndexReader(destination) as reader:
        assert reader.counts("sog_candidates", "train") == 1
        assert reader.metadata["prepared_baseline"]["fit"]["frames"] == 3
        atoms = reader.read_atoms(reader.record("sog_candidates", "test", 0))
        assert len(atoms) == 2


def test_cpu_help_exposes_portable_workflow():
    result = cpu_command("--help")
    assert all(name in result.stdout for name in ("seal", "bind", "dispatch"))


@pytest.mark.skipif(os.name != "nt", reason="exercise Windows PowerShell native argument passing on Windows CI")
def test_windows_powershell_preparation_wrapper(tmp_path):
    export = tmp_path / "export with spaces"
    exported_fixture(export, train=1, val=1, test=1)
    for shell in ("powershell.exe", "pwsh.exe"):
        if not shutil.which(shell):
            continue
        work = tmp_path / shell / "training"
        result = subprocess.run([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "examples/elementa/prepare_windows.ps1"), "-PythonExe", sys.executable,
            "-ExportDir", str(export), "-WorkDir", str(work)],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((work / "bundle/bundle.json").read_text(encoding="utf-8"))["status"] == "complete"
