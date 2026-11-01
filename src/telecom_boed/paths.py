"""[telecom_boed] Pinned locations of the τ²-bench dependency and per-run tau2 data dirs. Never imports tau2.

The tau2 clone is a read-only dependency. tau2 writes simulation results under TAU2_DATA_DIR
(`tau2/utils/utils.py:18-27`, read once at import), so every entry point copies `<clone>/data` into a scratch
dir and exports TAU2_DATA_DIR before the first tau2 import (see prepare_tau2_data_dir).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

TAU2_SHA = "4ce7c0397c1eb65c9bbe59aeacfe1ca44a1cd699"
# md5 of the shipped telecom task files; identical on Perlmutter (constraints/perlmutter_data_md5.txt).
TASKS_JSON_MD5 = "a610efdf3385d253f0601f2a4e3fc0ab"
SPLIT_TASKS_JSON_MD5 = "4a547c49d64d98ae3fe165acbfca1162"

_TAU2_ROOT_CANDIDATES = (
    Path("/common/users/ss5772/external/tau2-bench"),  # local (westeros / ilab / rlab, NFS)
    Path("/global/cfs/cdirs/m4431/kbmkbm/ext/tau2-bench"),  # Perlmutter CFS
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # .../telecom_boed
CONFIG_DIR = PACKAGE_ROOT / "configs"


def tau2_root() -> Path:
    """Pinned clone root: $TELECOM_TAU2_ROOT, else the first existing known location."""
    env = os.environ.get("TELECOM_TAU2_ROOT")
    if env:
        return Path(env)
    for p in _TAU2_ROOT_CANDIDATES:
        if (p / "src" / "tau2").is_dir():
            return p
    raise FileNotFoundError("tau2-bench clone not found; set TELECOM_TAU2_ROOT (scripts/setup_local.sh creates it)")


def clone_data_dir() -> Path:
    """The clone's read-only data dir (catalog source: tasks.json, split_tasks.json, db.toml, ...)."""
    return tau2_root() / "data"


def telecom_catalog_paths() -> tuple[Path, Path]:
    d = clone_data_dir() / "tau2" / "domains" / "telecom"
    return d / "tasks.json", d / "split_tasks.json"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_catalog_pins() -> None:
    """Fail loudly if the shipped telecom catalog differs from the pinned one (e.g. create_tasks.py was run)."""
    tasks, split = telecom_catalog_paths()
    got = (md5(tasks), md5(split))
    want = (TASKS_JSON_MD5, SPLIT_TASKS_JSON_MD5)
    if got != want:
        raise RuntimeError(f"telecom catalog md5 mismatch: got {got}, pinned {want}")


def prepare_tau2_data_dir(dest: Path) -> Path:
    """Copy `<clone>/data` to `dest` (once) and export TAU2_DATA_DIR=dest. Must run before any tau2 import."""
    if "tau2.utils.utils" in sys.modules:
        current = getattr(sys.modules["tau2.utils.utils"], "DATA_DIR", None)
        if current is None or Path(current).resolve() != Path(dest).resolve():
            raise RuntimeError(f"tau2 already imported with DATA_DIR={current}; set TAU2_DATA_DIR before importing tau2")
    dest = Path(dest)
    if not (dest / "tau2" / "domains" / "telecom" / "tasks.json").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(clone_data_dir(), dest, dirs_exist_ok=True)
    os.environ["TAU2_DATA_DIR"] = str(dest)
    return dest
