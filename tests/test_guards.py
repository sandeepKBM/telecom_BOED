"""[telecom_boed] Suite-level guards (owner: W0-O): pinned clone, catalog md5, data dir isolation, contract and knobs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from telecom_boed import contracts, paths


def test_clone_at_pinned_sha_and_clean():
    root = paths.tau2_root()
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == paths.TAU2_SHA
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert dirty.strip() == ""


def test_catalog_md5_pinned():
    paths.assert_catalog_pins()


def test_tau2_data_dir_outside_clone_and_used_by_tau2():
    data = Path(os.environ["TAU2_DATA_DIR"]).resolve()
    assert paths.tau2_root().resolve() not in data.parents
    from tau2.utils import utils as tau2_utils  # first tau2 import happens after conftest set TAU2_DATA_DIR

    assert Path(tau2_utils.DATA_DIR).resolve() == data


def test_contract_version():
    assert contracts.CONTRACT_VERSION == 1


def test_default_config_has_every_knob(boed_cfg):
    required = {
        "cost": {"lambda", "turn_cost", "tool_cost", "rho", "budget_B"},
        "utility": {"c_FA", "c_esc"},
        "reach_avoid": {"gamma", "delta", "eta", "p_star", "v_lr"},
        "lookahead": {"depth_k", "inner_m", "memo_round"},
        "user_model": {"compliance_q", "eps_double", "eps_extra", "default_mention_p"},
        "controller": {"n_candidates", "candidate_temperature", "override", "scorer"},
        "latency": {"p95_scoring_ms"},
    }
    for section, keys in required.items():
        assert keys <= set(boed_cfg[section]), (section, keys - set(boed_cfg[section]))
    assert boed_cfg["contract_version"] == contracts.CONTRACT_VERSION


def test_hashable_contract_types():
    a = contracts.DeviceAction("user", "toggle_airplane_mode", (), ("airplane_mode", False))
    o = contracts.Observation((a,), "user_report", contracts._frozen_items({"status_bar.signal": "none"}), frozenset({"status_bar.signal"}))
    assert hash(o.key()) == hash(o.key())
    h1 = contracts.Hypothesis(("mms_issue", frozenset({"a"})), ("t",), (), (), ("ENV_ASSERTION",), False, "g", False)
    h2 = contracts.Hypothesis(("mms_issue", frozenset({"a"})), ("u",), (), (), ("ENV_ASSERTION",), False, "g", False)
    assert h1 == h2 and len({h1, h2}) == 1
