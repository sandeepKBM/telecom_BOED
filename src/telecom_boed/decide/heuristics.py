"""[telecom_boed] Fixed-order heuristic baseline: follow the tau2 tech-support manual's check/repair order.

Order (tech_support_manual.md, tool names by section; line numbers are of the manual):
  service issues  (l.73-97):  check_status_bar, toggle_airplane_mode (74-77), check_sim_status, reseat_sim_card (79-86),
                               check_apn_settings, reset_apn_settings, reboot_device (88-91), [line suspension, agent side (93-98)]
  mobile data     (l.122-162): run_speed_test (123), check_network_status, toggle_airplane_mode (129), toggle_data (132),
                               toggle_roaming (136), check_data_restriction_status, toggle_data_saver_mode (142),
                               check_vpn_status, disconnect_vpn (146), [data plan limits, agent side (151-158)],
                               check_network_mode_preference, set_network_mode_preference (159-162)
  MMS             (l.164-209): can_send_mms (182), [basic connectivity (185)], set_network_mode_preference (189), check_apn_settings,
                               reset_apn_settings, reboot_device (194), check_wifi_calling_status, toggle_wifi_calling (200),
                               check_app_permissions, grant_app_permission (204)
MANUAL_ORDER is that list de-duplicated at first occurrence; tools not listed rank after it, in candidate order.
For toys the order is the toy spec's action order (pass `order=`).

The scorer is stateless in the belief, so the caller reports performed actions with `mark_done(cand)`. Score: nonterminal not done
-> -rank (<= 0); done -> -1e9; claim -> +1 if P(repaired) >= 1 - delta (cfg reach_avoid.delta) else -1e6; transfer -> -1e5 (so it is
taken only when everything is done and the claim is not justified). The heuristic never verifies beyond the manual's own checks.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from telecom_boed.contracts import Belief, CandidateAction, DecisionKind, Grader
from telecom_boed.decide.utility import p_repaired

MANUAL_ORDER: tuple[str, ...] = (
    "check_status_bar", "toggle_airplane_mode", "check_sim_status", "reseat_sim_card", "check_apn_settings", "reset_apn_settings",
    "reboot_device", "run_speed_test", "check_network_status", "toggle_data", "toggle_roaming", "check_data_restriction_status",
    "toggle_data_saver_mode", "check_vpn_status", "disconnect_vpn", "check_network_mode_preference", "set_network_mode_preference",
    "can_send_mms", "check_wifi_calling_status", "toggle_wifi_calling", "check_app_permissions", "grant_app_permission",
)


def cand_tool(c: CandidateAction) -> Optional[str]:
    return c.actions[0].tool if c.actions else None


class HeuristicScorer:
    name = "heuristic"

    def __init__(self, grader: Grader, cfg: Mapping[str, Any], order: Sequence[str] = MANUAL_ORDER):
        self.grader, self.cfg, self.order = grader, cfg, list(order)
        self.done: set[str] = set()

    def reset(self) -> None:
        self.done.clear()

    def mark_done(self, cand: CandidateAction) -> None:
        t = cand_tool(cand)
        if t is not None:
            self.done.add(t)

    def score(self, belief: Belief, candidates: Sequence[CandidateAction]) -> list[float]:
        ok = p_repaired(belief, self.grader) >= 1.0 - float(self.cfg["reach_avoid"]["delta"])
        out = []
        for pos, c in enumerate(candidates):
            if c.kind == DecisionKind.CLAIM_RESOLVED:
                out.append(1.0 if ok else -1e6)
            elif c.kind == DecisionKind.TRANSFER:
                out.append(-1e5)
            else:
                t = cand_tool(c)
                out.append(-1e9 if t in self.done else -float(self.order.index(t) if t in self.order else len(self.order) + pos))
        return out

    def decide(self, belief: Belief, candidates: Sequence[CandidateAction]) -> int:
        s = self.score(belief, candidates)
        return max(range(len(s)), key=lambda i: (s[i], -i))
