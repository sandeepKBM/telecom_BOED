"""[telecom_boed] Fast TelecomEnvironment builder + exact snapshot / restore / branch (owner: W1-S1). Imports tau2.

Restore = fresh env + direct assignment of both DBs + per-instance VPN default + sync_tools. Never
`set_state(initialization_data=...)` (it cross-wires the two DBs, environment.py:352-363).
State key = (tau2 agent-DB hash, tau2 user-DB hash, sha256 of the canonical JSON of the VPN default), so the
tau2-invisible VPN default (see tau2_patches H1) is part of identity.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.environment import TelecomEnvironment
from tau2.domains.telecom.tools import TelecomTools
from tau2.domains.telecom.user_data_model import TelecomUserDB, VpnDetails
from tau2.domains.telecom.user_tools import TelecomUserTools
from tau2.domains.telecom.utils import (
    TELECOM_DB_PATH,
    TELECOM_MAIN_POLICY_PATH,
    TELECOM_TECH_SUPPORT_POLICY_MANUAL_PATH,
    TELECOM_USER_DB_PATH,
)
from tau2.utils import load_file

from telecom_boed.contracts import Snapshot, StateKey

_POLICY: Optional[str] = None
_DEFAULT_AGENT_DB: Optional[dict] = None
_DEFAULT_USER_DB: Optional[dict] = None


def _defaults() -> tuple[str, dict, dict]:
    """Policy string and parsed default DB dicts, read from disk once per process (slow on NFS)."""
    global _POLICY, _DEFAULT_AGENT_DB, _DEFAULT_USER_DB
    if _POLICY is None:
        # Same text as telecom.environment.get_environment(policy_type="manual", solo_mode=False).
        _POLICY = (
            "<main_policy>\n" + load_file(TELECOM_MAIN_POLICY_PATH) + "\n</main_policy>\n"
            "<tech_support_policy>\n" + load_file(TELECOM_TECH_SUPPORT_POLICY_MANUAL_PATH) + "\n</tech_support_policy>"
        )
        _DEFAULT_AGENT_DB = TelecomDB.load(TELECOM_DB_PATH).model_dump()
        _DEFAULT_USER_DB = TelecomUserDB.load(TELECOM_USER_DB_PATH).model_dump()
    assert _DEFAULT_AGENT_DB is not None and _DEFAULT_USER_DB is not None
    return _POLICY, _DEFAULT_AGENT_DB, _DEFAULT_USER_DB


def fast_env(
    agent_db: Optional[Mapping[str, Any]] = None,
    user_db: Optional[Mapping[str, Any]] = None,
    vpn_default: Optional[Mapping[str, Any]] = None,
) -> TelecomEnvironment:
    """Equivalent to get_environment() (manual policy, not solo) without file IO. Dicts are validated into new
    models, never aliased. `vpn_default=None` keeps the pristine per-instance default."""
    policy, a0, u0 = _defaults()
    tools = TelecomTools(TelecomDB.model_validate(a0 if agent_db is None else agent_db))
    user_tools = TelecomUserTools(TelecomUserDB.model_validate(u0 if user_db is None else user_db))
    if vpn_default is not None:
        user_tools.default_vpn_details = VpnDetails.model_validate(vpn_default)
    return TelecomEnvironment(domain_name="telecom", policy=policy, tools=tools, user_tools=user_tools)


def _sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _vpn_dump(env: TelecomEnvironment) -> dict[str, Any]:
    return env.user_tools.default_vpn_details.model_dump(mode="json")


def state_key(env: TelecomEnvironment) -> StateKey:
    return (env.get_db_hash(), env.get_user_db_hash(), _sha(_vpn_dump(env)))


def snapshot(
    env: TelecomEnvironment, parent_id: Optional[str] = None, step: int = 0, messages: Sequence[Mapping[str, Any]] = ()
) -> Snapshot:
    key = state_key(env)
    return Snapshot(
        agent_db=env.tools.db.model_dump(),
        user_db=env.user_tools.db.model_dump(),
        vpn_default=_vpn_dump(env),
        key=key,
        snapshot_id=_sha([key, parent_id, step])[:12],
        parent_id=parent_id,
        step=step,
        messages=tuple(copy.deepcopy(dict(m)) for m in messages),
    )


def snapshot_reusing(env: TelecomEnvironment, parent: Snapshot) -> Snapshot:
    """snapshot(env) with the three state-hash parts reused from `parent` when the corresponding dump is unchanged (P0-2, perf.reuse_state_hash).
    Dict equality is a C-level walk; tau2 get_db_hash is json.dumps(sort_keys)+sha256 of the same dump, ~half the cost of a cache-miss
    apply_calls. Equal dumps give equal hashes, so key and snapshot_id are bit-identical to snapshot()'s (tests/test_perf_v2.py).
    Lives here, not in env/particles.py, because the leakage scan forbids the user-DB attributes there."""
    agent_db, user_db, vpn = env.tools.db.model_dump(), env.user_tools.db.model_dump(), _vpn_dump(env)
    k0 = parent.key[0] if agent_db == parent.agent_db else env.get_db_hash()
    k1 = parent.key[1] if user_db == parent.user_db else env.get_user_db_hash()
    k2 = parent.key[2] if vpn == parent.vpn_default else _sha(vpn)
    key = (k0, k1, k2)
    return Snapshot(agent_db=agent_db, user_db=user_db, vpn_default=vpn, key=key, snapshot_id=_sha([key, None, 0])[:12],
                    parent_id=None, step=0, messages=())


def restore(snap: Snapshot) -> TelecomEnvironment:
    """Fresh env in exactly the snapshot's state; shares no mutable object with `snap`."""
    env = fast_env(snap.agent_db, snap.user_db, snap.vpn_default)
    env.sync_tools()
    dev, ut = env.user_tools.device, env.user_tools
    # In tau2 a connected device's vpn_details IS the instance default object (connect_vpn aliases it), and
    # break_vpn relies on that. Re-establish the alias when the dumps agree.
    if dev.vpn_connected and dev.vpn_details is not None and (
        dev.vpn_details.model_dump(mode="json") == _vpn_dump(env)
    ):
        dev.vpn_details = ut.default_vpn_details
    return env


def branch(snap: Snapshot, n: int) -> list[TelecomEnvironment]:
    return [restore(snap) for _ in range(n)]
