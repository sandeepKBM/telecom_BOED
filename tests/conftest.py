"""[telecom_boed] pytest bootstrap (owner: W0-O). This file runs before any test module imports tau2.

It does four things:
1. Puts pytest's basetemp on node-local disk, not /tmp (node /tmp quota is about 2 GB).
2. Copies the pinned clone's data/ to a scratch dir and exports TAU2_DATA_DIR, so tau2 never writes into the clone.
3. Applies the tau2 process patches once W1-S1 has created env/tau2_patches.py.
4. Fails the session if the clone ends up dirty or the catalog md5 changes.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from telecom_boed import paths

_SCRATCH_CANDIDATES = (Path("/data/local/ss5772/tmp"), Path("/freespace/local/ss5772/tmp"))


def _scratch_root() -> Path:
    env = os.environ.get("TELECOM_SCRATCH")
    if env:
        return Path(env)
    for p in _SCRATCH_CANDIDATES:
        if p.is_dir() and os.access(p, os.W_OK):
            return p / "telecom_boed"
    return paths.PACKAGE_ROOT / ".scratch"  # NFS fallback; still never /tmp


SCRATCH = _scratch_root()
TAU2_TEST_DATA = paths.prepare_tau2_data_dir(SCRATCH / f"tau2_data_{paths.TAU2_SHA[:8]}")

if (Path(paths.__file__).parent / "env" / "tau2_patches.py").exists():
    from telecom_boed.env import tau2_patches  # noqa: E402

    tau2_patches.apply()


def pytest_configure(config: pytest.Config) -> None:
    if not config.option.basetemp:
        config.option.basetemp = str(SCRATCH / f"pytest_{os.getpid()}")


def _clone_dirty() -> str:
    root = paths.tau2_root()
    out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout
    return out.strip()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    problems = []
    if _clone_dirty():
        problems.append("tau2 clone has uncommitted changes (a test wrote into the clone)")
    try:
        paths.assert_catalog_pins()
    except RuntimeError as e:
        problems.append(str(e))
    if problems:
        print("\n[telecom_boed guard] " + "; ".join(problems))
        session.exitstatus = 1


# --------------------------------------------------------------------------- shared fixtures
@pytest.fixture(scope="session")
def boed_cfg() -> dict:
    with open(paths.CONFIG_DIR / "boed_default.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def catalog() -> list[dict]:
    tasks, _ = paths.telecom_catalog_paths()
    with open(tasks) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def splits() -> dict[str, list[str]]:
    _, split = paths.telecom_catalog_paths()
    with open(split) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def train_ids(splits) -> list[str]:
    return list(splits["train"])


@pytest.fixture(scope="session")
def test40_ids(splits) -> frozenset[str]:
    """Only split/leakage tests may use this. Never use test-40 tasks as inputs elsewhere."""
    return frozenset(splits["test"])
