"""[telecom_boed] Process-level patches for two tau2-bench telecom hazards (owner: W1-S1). Imports tau2.

Call `apply()` once per process, after TAU2_DATA_DIR is set and before any environment is built (conftest does).

H1 VPN alias. `TelecomUserTools.default_vpn_details` is a CLASS attribute and `connect_vpn` stores that very
   object in `device.vpn_details`; `break_vpn` then mutates it, so one broken env poisons every later env in the
   process. tau2's DB hashes do not see it. Patch: `__init__` gives each instance its own deep copy of the
   pristine default (instance attribute shadows the class one). Aliasing device<->default inside one instance is
   kept on purpose: it is what makes "break_vpn, disconnect, reconnect" stay broken in tau2.
H2 Random bill ids. `_apply_one_time_charge` builds `B{uuid.uuid4().hex[:8]}` for a customer with no draft bill.
   Patch: the module's `uuid` is replaced by a shim whose uuid4() is a pure function of (customer_id, number of
   bills already in the DB) (+ a collision salt). The id depends only on DB content, so live runs, snapshot
   restores and `set_state` replays all mint the same id. A process-wide counter would not survive restore.
"""
from __future__ import annotations

import hashlib
import os
import threading
import uuid as _real_uuid
from functools import wraps
from typing import Any

PATCHES: list[str] = [
    "H1: TelecomUserTools.__init__ gives each instance a private deep copy of the pristine default_vpn_details "
    "(class-attribute alias corrupted by break_vpn)",
    "H2: tau2.domains.telecom.tools.uuid -> shim; bill id = sha256(customer_id|n_bills[|salt]) so "
    "_apply_one_time_charge is deterministic and a function of DB content only",
]

_PRISTINE_VPN: dict[str, Any] | None = None
_ctx = threading.local()
_APPLIED = False


class _DeterministicUuid:
    """Stands in for the `uuid` module inside tau2.domains.telecom.tools; only uuid4 is context-dependent."""

    def uuid4(self) -> _real_uuid.UUID:
        seed = getattr(_ctx, "bill_seed", None)
        if seed is None:
            raise RuntimeError("uuid4() called outside _apply_one_time_charge; unexpected new randomness in tau2")
        return _real_uuid.UUID(hex=hashlib.sha256(seed.encode()).hexdigest()[:32])

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_uuid, name)


def pristine_vpn() -> dict[str, Any]:
    """JSON-mode dump of the default VpnDetails as captured at apply() time (a fresh dict each call)."""
    if _PRISTINE_VPN is None:
        raise RuntimeError("tau2_patches.apply() has not run")
    return dict(_PRISTINE_VPN)


def apply() -> None:
    """Install the patches (idempotent). Refuses to run unless TAU2_DATA_DIR is set."""
    global _APPLIED, _PRISTINE_VPN
    if not os.environ.get("TAU2_DATA_DIR"):
        raise RuntimeError("TAU2_DATA_DIR is unset: tau2 would write into the clone's data dir; set it first")
    if _APPLIED:
        return
    from tau2.domains.telecom import tools as tel_tools
    from tau2.domains.telecom.user_data_model import PerformanceLevel, VpnDetails
    from tau2.domains.telecom.user_tools import TelecomUserTools

    pristine = TelecomUserTools.default_vpn_details
    if (pristine.server_address, pristine.protocol, pristine.server_performance) != (
        "192.168.1.1", "OpenVPN", PerformanceLevel.EXCELLENT,
    ):
        raise RuntimeError(f"default_vpn_details already corrupted before apply(): {pristine!r}")
    _PRISTINE_VPN = pristine.model_dump(mode="json")

    orig_init = TelecomUserTools.__init__

    @wraps(orig_init)
    def init(self: Any, db: Any) -> None:
        orig_init(self, db)
        self.default_vpn_details = VpnDetails.model_validate(_PRISTINE_VPN)

    TelecomUserTools.__init__ = init

    orig_charge = tel_tools.TelecomTools._apply_one_time_charge

    @wraps(orig_charge)
    def charge(self: Any, customer_id: str, *args: Any, **kwargs: Any) -> Any:
        existing = {b.bill_id for b in self.db.bills}
        seed = f"{customer_id}|{len(self.db.bills)}"
        salt = 0
        while f"B{hashlib.sha256(seed.encode()).hexdigest()[:8]}" in existing:
            salt += 1
            seed = f"{customer_id}|{len(self.db.bills)}|{salt}"
        prev = getattr(_ctx, "bill_seed", None)
        _ctx.bill_seed = seed
        try:
            return orig_charge(self, customer_id, *args, **kwargs)
        finally:
            _ctx.bill_seed = prev

    tel_tools.TelecomTools._apply_one_time_charge = charge
    tel_tools.uuid = _DeterministicUuid()
    _APPLIED = True
