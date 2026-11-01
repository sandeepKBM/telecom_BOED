"""[telecom_boed] User model for dual control + tool semantics tables (owner: W2-S5). Deployable; no tau2 import.

Dual control: the agent never sees phone-tool output. It sees only what the user chooses to say. The model below says
(a) whether the user does what was asked (execution branches), and (b) what they report (mention + confusion).

Execution branches of one REQUEST_DEVICE action (each requested action independently):
  not_done   1 - q - eps_double   the tool never ran
  done       q                    ran once
  done_twice eps_double           ran twice (matters for toggles: net effect is undone)
  plus, per request, with prob eps_extra the user also runs ONE unrequested mutating action, `reboot_device`
  (EXTRA_ACTION): a common over-helpful step; it resets the APN to the broken default (user_tools.py:955-958).
Report model (per output group, see belief/filter.py for how it enters the likelihood):
  P(group mentioned) = min(1, default_mention_p * boost[group]);  given mentioned, the reported tuple is
  (1 - eps_conf) * [== truth] + eps_conf * Uniform(label space);  if the tool never ran: Uniform(label space)
  (`report_when_not_done: uniform`) or the truth the tool WOULD have shown (`claims_done`).
All numbers are PLACEHOLDER PRIORS read from configs/boed_default.yaml `user_model:` plus configs/user_model/*.yaml; they are
to be fitted on train-split logs. Nothing here reads a task or a user DB.

Tool tables (all line numbers: tau2 `src/tau2/domains/telecom/user_tools.py` (uf) and `tools.py` (tf)):
READ_TOOLS = tools registered `@is_tool(ToolType.READ)` (uf:76,142,179,246,417,529,576,630,675,718,794,804,835,905,1054;
  tf:48,72,89,233,334,484). Every other tool is WRITE or GENERIC, i.e. "mutating" for the v rule (contracts.py docstring).
PRIMARY_GROUPS = output groups the user is expected to report, used for the lookahead observation space in `predict`.
  Group = prefix of the featurize() field name. Justification, by tool:
  - check_status_bar (uf:76-139), and every write tool whose reply ends with `Status Bar: ...` (toggle_airplane_mode uf:366-377,
    toggle_data uf:471, toggle_roaming uf:499, toggle_data_saver_mode uf:546, toggle_wifi uf:650-657, toggle_wifi_calling uf:690,
    connect_vpn uf:745-756, disconnect_vpn uf:768-777, reseat_sim_card uf:435-439, set_network_mode_preference uf:188,
    set_apn_settings uf:590-597, reset_apn_settings uf:604-609, reboot_device uf:941-964): the phone shows the bar after the
    action and the bar is the state evidence (airplane / signal / tech / data / wifi / vpn icons), so the primary group is
    `status_bar`. The write tool's own status sentence (`<tool>.msg`) is an echo of what the user did, so it is not reportable.
  - read tools: their own group (network uf:142-159, network_mode uf:179, speed_test uf:246-270, sim uf:417-427,
    data_restriction uf:529-537, apn uf:576-583, wifi uf:630-639, wifi_calling uf:675-681, vpn uf:718-731, apps uf:794-798,
    app_status uf:804-826, app_permissions uf:835-851, mms uf:905-912, payment_request uf:1054-1062).
  - grant_app_permission (uf:860-870): the one fact the user reports is `Success.` vs `Error.` -> group `app_permission`.
  - make_payment (uf:1072-1090): `make_payment`.
  A tool not listed reports every group of its output (computed by the filter).
PRIMARY_READ[group] = read-only tool that shows the group's current value. It supplies (a) the report when an `ensure` is
  already satisfied and the user just looks at the screen, (b) the ensure value reads (ENSURE_READ).
ENSURE_READ[field] = (read tool, featurized field) giving the current value of a DeviceAction.ensure field. An `ensure`
  field not listed here is ignored (the action's tool always runs).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping, Optional

import yaml

from telecom_boed import paths
from telecom_boed.contracts import DeviceAction

EXTRA_ACTION = DeviceAction("user", "reboot_device")  # the unrequested mutating action of the `eps_extra` branch

READ_TOOLS = frozenset({
    "check_status_bar", "check_network_status", "check_network_mode_preference", "run_speed_test", "check_sim_status",
    "check_data_restriction_status", "check_apn_settings", "check_wifi_status", "check_wifi_calling_status",
    "check_vpn_status", "check_installed_apps", "check_app_status", "check_app_permissions", "can_send_mms",
    "check_payment_request",
    "get_customer_by_phone", "get_customer_by_id", "get_customer_by_name", "get_details_by_id",
    "get_bills_for_customer", "get_data_usage",
})

_BAR_TOOLS = ("toggle_airplane_mode", "toggle_data", "toggle_roaming", "toggle_data_saver_mode", "toggle_wifi",
              "toggle_wifi_calling", "connect_vpn", "disconnect_vpn", "reseat_sim_card", "set_network_mode_preference",
              "set_apn_settings", "reset_apn_settings", "reboot_device")

PRIMARY_GROUPS: dict[str, tuple[str, ...]] = {
    "check_status_bar": ("status_bar",),
    **{t: ("status_bar",) for t in _BAR_TOOLS},
    "check_network_status": ("network",),
    "check_network_mode_preference": ("network_mode",),
    "run_speed_test": ("speed_test",),
    "check_sim_status": ("sim",),
    "check_data_restriction_status": ("data_restriction",),
    "check_apn_settings": ("apn",),
    "check_wifi_status": ("wifi",),
    "check_wifi_calling_status": ("wifi_calling",),
    "check_vpn_status": ("vpn",),
    "check_installed_apps": ("apps",),
    "check_app_status": ("app_status",),
    "check_app_permissions": ("app_permissions",),
    "can_send_mms": ("mms",),
    "check_payment_request": ("payment_request",),
    "grant_app_permission": ("app_permission",),
    "make_payment": ("make_payment",),
}

PRIMARY_READ: dict[str, str] = {
    "status_bar": "check_status_bar", "network": "check_network_status", "network_mode": "check_network_mode_preference",
    "speed_test": "run_speed_test", "sim": "check_sim_status", "data_restriction": "check_data_restriction_status",
    "apn": "check_apn_settings", "wifi": "check_wifi_status", "wifi_calling": "check_wifi_calling_status",
    "vpn": "check_vpn_status", "apps": "check_installed_apps", "mms": "can_send_mms",
    "payment_request": "check_payment_request",
}

ENSURE_READ: dict[str, tuple[str, str]] = {
    "airplane_mode": ("check_status_bar", "status_bar.airplane_mode"),
    "data": ("check_network_status", "network.data_enabled"),
    "roaming": ("check_network_status", "network.roaming_enabled"),
    "wifi": ("check_network_status", "network.wifi_enabled"),
    "data_saver": ("check_data_restriction_status", "data_restriction.data_saver"),
    "wifi_calling": ("check_wifi_calling_status", "wifi_calling.enabled"),
    "vpn": ("check_vpn_status", "vpn.connected"),
}


def is_mutating_tool(tool: str) -> bool:
    """True for WRITE / GENERIC tools (anything not registered ToolType.READ)."""
    return tool not in READ_TOOLS


def group_of(field_name: str) -> str:
    return field_name.split(".", 1)[0]


def reportable(field_name: str) -> bool:
    """Free-text status sentences (`.msg`) and unparsed text (`.raw`) are not something a user reports field-by-field."""
    return not (field_name.endswith(".msg") or field_name.endswith(".raw"))


def primary_groups(tool: str) -> Optional[tuple[str, ...]]:
    return PRIMARY_GROUPS.get(tool)


# --------------------------------------------------------------------------- parameters
def _opt(v: Any) -> Optional[float]:
    return None if v is None else float(v)


def _deep_merge(base: dict, over: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@lru_cache(maxsize=None)
def _read_yaml(name: str) -> dict:
    p = paths.CONFIG_DIR / "user_model" / f"{name}.yaml"
    with open(p) as f:
        return yaml.safe_load(f) or {}


def available_personas() -> list[str]:
    return sorted(p.stem for p in (paths.CONFIG_DIR / "user_model").glob("*.yaml"))


@dataclass(frozen=True)
class UserModel:
    persona: str = "default"
    q: float = 0.9
    eps_double: float = 0.02
    eps_extra: float = 0.02
    default_mention_p: float = 0.7
    eps_conf: float = 0.05
    report_when_not_done: str = "uniform"
    mention_boost: Mapping[str, float] = field(default_factory=dict)
    # Optional per-tool-type compliance (cfg compliance_q_read / compliance_q_mutating). None = use q (the old behaviour).
    q_read: Optional[float] = None
    q_mutating: Optional[float] = None

    def __post_init__(self) -> None:
        for n, v in (("q", self.q), ("eps_double", self.eps_double), ("eps_extra", self.eps_extra),
                     ("default_mention_p", self.default_mention_p), ("eps_conf", self.eps_conf),
                     ("q_read", self.q_read), ("q_mutating", self.q_mutating)):
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"user_model.{n}={v} outside [0, 1]")
        for n, v in (("compliance_q", self.q), ("compliance_q_read", self.q_read), ("compliance_q_mutating", self.q_mutating)):
            if v is not None and v + self.eps_double > 1.0 + 1e-12:
                raise ValueError(f"user_model: {n} + eps_double = {v + self.eps_double} > 1")
        if self.report_when_not_done not in ("uniform", "claims_done"):
            raise ValueError(f"report_when_not_done={self.report_when_not_done!r}: expected uniform | claims_done")

    @property
    def p_not_done(self) -> float:
        return max(0.0, 1.0 - self.q - self.eps_double)

    def branch_probs(self) -> dict[str, float]:
        """Per-action execution branch probabilities (sum to 1)."""
        return {"not_done": self.p_not_done, "done": self.q, "done_twice": self.eps_double}

    def q_for(self, tool: str) -> float:
        """Compliance for one requested tool: q_mutating / q_read when set, else q."""
        v = self.q_mutating if is_mutating_tool(tool) else self.q_read
        return self.q if v is None else v

    def branches_for(self, tool: str) -> tuple[tuple[str, float], ...]:
        """(nd, d, dd) branch probabilities for one requested tool (filter._combos). Equals (p_not_done, q, eps_double) when
        the per-type values are unset."""
        q = self.q_for(tool)
        return (("nd", max(0.0, 1.0 - q - self.eps_double)), ("d", q), ("dd", self.eps_double))

    def mention_p(self, group: str) -> float:
        return min(1.0, self.default_mention_p * float(self.mention_boost.get(group, 1.0)))

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any], persona: str = "default") -> "UserModel":
        """boed_default `user_model:` block < user_model/default.yaml < user_model/<persona>.yaml (deep merge, later wins).
        Persona "default" means: no persona file beyond default.yaml."""
        base = dict(cfg.get("user_model", {}))
        merged = {"compliance_q": base.get("compliance_q", 0.9), "eps_double": base.get("eps_double", 0.02),
                  "eps_extra": base.get("eps_extra", 0.02), "default_mention_p": base.get("default_mention_p", 0.7)}
        merged = _deep_merge(merged, _read_yaml("default"))
        if persona != "default":
            if persona not in available_personas():
                raise KeyError(f"unknown persona {persona!r}; have {available_personas()}")
            merged = _deep_merge(merged, _read_yaml(persona))
        return cls(persona=persona, q=float(merged["compliance_q"]), eps_double=float(merged["eps_double"]),
                   eps_extra=float(merged["eps_extra"]), default_mention_p=float(merged["default_mention_p"]),
                   eps_conf=float(merged.get("eps_conf", 0.05)),
                   report_when_not_done=str(merged.get("report_when_not_done", "uniform")),
                   mention_boost=dict(merged.get("mention_boost", {})),
                   q_read=_opt(base.get("compliance_q_read", merged.get("compliance_q_read"))),
                   q_mutating=_opt(base.get("compliance_q_mutating", merged.get("compliance_q_mutating"))))
