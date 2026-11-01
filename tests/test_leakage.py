"""[telecom_boed] Leakage scanner + runtime poison tests (owner: W1-S3)."""
from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pytest

from telecom_boed.testing.leakage_scan import (
    DEPLOYABLE_MODULES, check_deployable, scan_module_source, transitive_imports)
from telecom_boed.testing.poison import (
    LEAK_LOG, LeakageError, PoisonDB, PoisonTask, assert_identical_under_poison, poisoned)

FIX = Path(__file__).parent / "fixtures" / "W1-S3" / "pkg"
PKGDIR = FIX / "telecom_boed"


def _kinds(name: str) -> set[str]:
    return {v.kind for v in scan_module_source(PKGDIR / f"{name}.py")}


def test_scanner_flags_synthetic_violations():
    assert "forbidden_attr" in _kinds("bad_task")
    assert "forbidden_attr" in _kinds("bad_userdb") and "forbidden_str" in _kinds("bad_userdb")
    assert {"oracle_import", "task_read", "forbidden_str", "dynamic"} <= _kinds("bad_misc")
    assert scan_module_source(PKGDIR / "clean.py") == []
    # transitive: own source of bad_transitive is clean, but its closure reaches oracle_hidden via helper
    assert scan_module_source(PKGDIR / "bad_transitive.py") == []
    assert "telecom_boed.oracle_hidden" in transitive_imports("telecom_boed.bad_transitive", FIX)
    names = ["clean", "bad_task", "bad_transitive", "bad_userdb", "bad_misc", "not_there"]
    rep = check_deployable([f"telecom_boed.{n}" for n in names], FIX)
    assert rep.pending == ["telecom_boed.not_there"]
    flagged = {m.split(".")[-1] for m, r in rep.modules.items() if not r.clean}
    assert flagged == {"bad_task", "bad_transitive", "bad_userdb", "bad_misc"}
    chain = rep.modules["telecom_boed.bad_transitive"].oracle_chains["telecom_boed.oracle_hidden"]
    assert chain == ["telecom_boed.bad_transitive", "telecom_boed.helper", "telecom_boed.oracle_hidden"]
    assert not rep.ok and rep.modules["telecom_boed.clean"].clean


def test_transitive_imports_include_parent_packages():
    assert {"telecom_boed", "telecom_boed.paths"} <= transitive_imports("telecom_boed.clean", FIX)


def test_deployable_modules_clean():
    assert len(DEPLOYABLE_MODULES) == 6  # + tau2_ext.belief_script, tau2_ext.tool_mask (2026-11-01)
    rep = check_deployable(DEPLOYABLE_MODULES)
    print(f"\n[W1-S3] pending deployable modules: {rep.pending}; scanned: {sorted(rep.modules)}")
    assert not rep.problems(), "\n".join(rep.problems())


def test_poison_raises():
    for cls in (PoisonTask, PoisonDB):
        p = cls()
        assert repr(p) == f"<{cls.__name__}>" and p.__class__ is cls and isinstance(p, cls)
        accesses = [
            lambda: p.initial_state, lambda: p.user_db, lambda: p.__dict__, lambda: p.__doc__,
            lambda: p["k"], lambda: iter(p), lambda: list(p), lambda: len(p), lambda: "k" in p,
            lambda: str(p), lambda: f"{p}", lambda: bool(p), lambda: hash(p), lambda: p == 1, lambda: p != p,
            lambda: p(), lambda: p + 1, lambda: 1 + p, lambda: int(p), lambda: dir(p),
            lambda: copy.copy(p), lambda: copy.deepcopy(p), lambda: pickle.dumps(p),
            lambda: setattr(p, "x", 1), lambda: p.__setitem__("k", 1), lambda: getattr(p, "task", None),
            lambda: hasattr(p, "x"),
        ]
        for i, f in enumerate(accesses):
            n = len(LEAK_LOG)
            with pytest.raises(LeakageError):
                f()
            assert len(LEAK_LOG) == n + 1, f"access #{i} not logged for {cls.__name__}"
    # LeakageError is not an Exception subclass, so `except Exception` cannot swallow it
    assert not issubclass(LeakageError, Exception)


def test_poisoned_detects_swallowed_leak_and_identical_output():
    def agent(task=None, user_db=None, **kw):
        del task, user_db
        return ("ok", sorted(kw))

    assert assert_identical_under_poison(agent, {"task": object(), "user_db": object(), "a": 1}, {"task": PoisonTask, "user_db": PoisonDB}) == ("ok", ["a"])

    def leaky(task=None):
        try:
            task.initial_state
        except BaseException:
            pass
        return 1

    with pytest.raises(LeakageError, match="swallowed"):
        with poisoned(task=PoisonTask) as p:
            leaky(**p)
