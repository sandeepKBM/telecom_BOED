"""[telecom_boed] Risk table R(a) and kappa (docs/proposal.md §3, eq. 2' term rho*R(a); contracts.py "Utility semantics").

`risk(cand, belief) -> (R, kappa)`: R in [0, 1] is the weight multiplied by rho in the DUG. kappa is True when the action may
violate the tau2 policy or be irreversible. Transfer of a fixable fault is NOT handled here (utility semantics do that).
Unknown actions return (0.0, False). Table-driven; ground truth for the rows is tau2 domains/telecom/tools.py and main_policy.md.

Rows (backend = requestor "assistant"; catalog has one customer, C1001, line L1002):
- resume_line: policy (main_policy.md:126) forbids lifting a suspension when the line's contract_end_date is in the past;
  tools.py:298-330 resumes without checking. R = belief mass of particles where that holds for the target line; kappa = mass > 0
  (conservative: one possible violating particle blocks the action under reach-avoid).
- suspend_line / suspend_line_for_overdue_bill: customer-visible loss of service, reversible via resume_line: R = 1.0, kappa False.
- send_payment_request: the tool does not check the bill is unpaid/overdue (tools.py:361-370): R = 0.5 if the target bill is not
  Overdue in some particle (mass-weighted), kappa False.
- any other write tool on a customer_id or line_id other than the identified one (C1001 / L1002): R = 1.0, kappa True.
- set_data_usage / refuel_data / enable_roaming / disable_roaming on the right line: R = 0.1 (state-changing, billing-relevant, reversible), kappa False.
- user-side actions (phone) are all reversible: R = 0.0 (reset_apn_settings, toggles, reboot_device are fine).
"""
from __future__ import annotations

from typing import Any

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind

KNOWN_CUSTOMER = "C1001"
KNOWN_LINE = "L1002"
TODAY = "2025-02-25"  # tau2 telecom utils.get_today(); ISO strings compare lexicographically

SIMPLE_RISK: dict[str, tuple[float, bool]] = {
    "suspend_line": (1.0, False),
    "suspend_line_for_overdue_bill": (1.0, False),
    "set_data_usage": (0.1, False),
    "refuel_data": (0.1, False),
    "enable_roaming": (0.1, False),
    "disable_roaming": (0.1, False),
}
WRITE_TOOLS = frozenset(SIMPLE_RISK) | {"resume_line", "send_payment_request"}


def _find(db: Any, table: str, key: str, val: Any) -> list[dict]:
    rows = db.get(table, []) if hasattr(db, "get") else []
    return [r for r in rows if isinstance(r, dict) and r.get(key) == val]


def _resume_violation_mass(belief: Belief, line_id: str) -> float:
    m = 0.0
    for p in belief.particles:
        for ln in _find(p.state.agent_db, "lines", "line_id", line_id):
            end = ln.get("contract_end_date")
            if str(ln.get("status")) in ("Suspended", "LineStatus.SUSPENDED") and end is not None and str(end) < TODAY:
                m += p.weight
    return m


def _bill_not_overdue_mass(belief: Belief, bill_id: str) -> float:
    m = 0.0
    for p in belief.particles:
        bills = _find(p.state.agent_db, "bills", "bill_id", bill_id)
        if not bills or any(str(b.get("status")) not in ("Overdue", "BillStatus.OVERDUE") for b in bills):
            m += p.weight
    return m


def risk(cand: CandidateAction, belief: Belief) -> tuple[float, bool]:
    """R(a) and kappa for a candidate: max over its sub-actions (R) and OR (kappa)."""
    if cand.kind in (DecisionKind.CLAIM_RESOLVED, DecisionKind.TRANSFER, DecisionKind.ASK):
        return 0.0, False
    r_tot, k_tot = 0.0, False
    for a in cand.actions:
        if a.requestor != "assistant" or a.tool not in WRITE_TOOLS:
            continue
        args = a.args_dict()
        cust, line = args.get("customer_id"), args.get("line_id")
        if (cust is not None and cust != KNOWN_CUSTOMER) or (line is not None and line != KNOWN_LINE):
            r, k = 1.0, True
        elif a.tool == "resume_line":
            mass = _resume_violation_mass(belief, line or KNOWN_LINE)
            r, k = mass, mass > 0.0
        elif a.tool == "send_payment_request":
            r, k = 0.5 * _bill_not_overdue_mass(belief, args.get("bill_id")), False
        else:
            r, k = SIMPLE_RISK[a.tool]
        r_tot, k_tot = max(r_tot, r), k_tot or k
    return r_tot, k_tot
