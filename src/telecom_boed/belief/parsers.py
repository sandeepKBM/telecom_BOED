"""[telecom_boed] Text parsers for the Bayes filter (owner: W2-S7). Rule-based first; LLM-backed stubs share the protocols.

Why this exists: in tau2 telecom dual control the agent has 13 backend tools; the phone's 30 tools are run by the
simulated user and the agent only ever sees the user's TEXT reply. The filter therefore needs four parsers, and their
error is its own error source (measured by `telecom_boed.eval.parser_accuracy`):

1. `RuleActionParser`     agent text -> which phone actions it requested (`DeviceAction`, with `ensure` for toggles).
2. `RuleObservationParser` user text -> featurize()-named fields + values, restricted to a CLOSED label space.
3. `RuleClaimClassifier`  agent text -> P(the message claims the issue is resolved) (the "commit" r).
4. `RuleIntentParser`     first user text -> {service_issue, mobile_data_issue, mms_issue} probabilities.

Deployable module: parses TEXT only. It never imports an `oracle_*` module, never touches the user's tool results,
and never reads the episode's task or databases (the leakage scanner enforces this on this file).

Conventions fixed here (contracts.py is not edited)
- `ensure` field names are the toggle groups of env.featurize: ("airplane_mode"|"data"|"roaming"|"data_saver"|"wifi"|
  "wifi_calling", bool) for the six toggles, ("vpn", bool) for connect_vpn / disconnect_vpn. `ensure=None` when the
  agent says only "toggle it" without a direction.
- Conditionals: `parse_all()` returns every request with a `conditional` flag. `parse()` returns the unconditional
  requests in textual order; if the message has only conditional ones it returns the first of them (the user will most
  likely act on it). A sentence is conditional when it starts with "if ..."/"otherwise ..."/"unless ...", or when the
  text after a mid-sentence "if/otherwise" asks for something, or when it ends in a failure-type condition ("..., if that
  doesn't work"). The condition clause itself ("If you see airplane mode") is never parsed as a request.
- Negated requests ("you don't need to restart") are dropped entirely.
- Conflicts: when the user's text says two different things about one fact in the present tense ("airplane mode is on ...
  actually it's off"), no value is emitted and the field is NOT in `mentioned` (a null report never moves the filter).
  The conflict is surfaced on `ParsedObservation` (a subclass of `Observation`, so it is accepted anywhere an
  `Observation` is): `.conflicts` = ((field, (values...)), ...) and `.out_of_space` = ((field, value), ...) for values
  the closed label space does not allow. `mentioned` is therefore always a subset of the field names in `fields`.
- Past-tense clauses ("it was on", "earlier", "before") are ignored; the post-action state is what is reported.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from telecom_boed.contracts import (INTENTS, DecisionKind, DeviceAction, Observation, _frozen_items)

USER_TOOL_NAMES: tuple[str, ...] = (
    "check_status_bar", "check_network_status", "check_network_mode_preference", "set_network_mode_preference",
    "run_speed_test", "toggle_airplane_mode", "check_sim_status", "reseat_sim_card", "toggle_data", "toggle_roaming",
    "check_data_restriction_status", "toggle_data_saver_mode", "check_apn_settings", "set_apn_settings",
    "reset_apn_settings", "check_wifi_status", "toggle_wifi", "check_wifi_calling_status", "toggle_wifi_calling",
    "check_vpn_status", "connect_vpn", "disconnect_vpn", "check_installed_apps", "check_app_status",
    "check_app_permissions", "grant_app_permission", "can_send_mms", "reboot_device", "check_payment_request",
    "make_payment")
NETWORK_MODES: tuple[str, ...] = ("4g_5g_preferred", "4g_only", "3g_only", "2g_only")  # NetworkModePreference values


# =========================================================================== text normalisation
def _norm(text: Optional[str]) -> str:
    t = (text or "").lower()
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("✈️", "airplane mode "),
                 ("✈", "airplane mode ")):
        t = t.replace(a, b)
    t = re.sub(r"[*`#>]+", " ", t)
    t = re.sub(r"wi[\s-]?fi", "wifi", t)
    t = re.sub(r"\b(?:flight|aeroplane|air plane)\s+mode\b", "airplane mode", t)
    t = re.sub(r"\bre-(seat|insert)", r"re\1", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _sentence_spans(t: str) -> list[tuple[int, int]]:
    """Sentence spans: split after .!? followed by whitespace, at newlines and at semicolons (decimals survive)."""
    cuts = [0] + [m.end() for m in re.finditer(r"(?<=[.!?;])\s+|\n+", t)] + [len(t)]
    return [(a, b) for a, b in zip(cuts, cuts[1:]) if t[a:b].strip()]


# =========================================================================== action parser
@dataclass(frozen=True)
class ParsedAction:
    """One requested phone action. `conditional` = only requested under a condition ("if that doesn't work, ...")."""
    action: DeviceAction
    conditional: bool
    start: int  # offset in the normalised text (textual order key)
    sentence: str


_OBJ = re.compile(
    r"(?P<wifi_calling>wifi calling|wifi calls?)|(?P<data_saver>data sav(?:er|ing)(?: mode)?)|"
    r"(?P<roaming>(?:data )?roaming)|(?P<airplane_mode>airplane(?: mode| icon| symbol)?)|"
    r"(?P<wifi>wifi(?: radio| connection)?)|(?P<vpn>\bvpn\b)|"
    r"(?P<data>(?:mobile|cellular) data|data connection|\bdata\b(?!\s+(?:saver|saving|roaming|usage|plan|limit|used|allowance)))")
_TOGGLE_TOOL = {"airplane_mode": "toggle_airplane_mode", "wifi_calling": "toggle_wifi_calling", "wifi": "toggle_wifi",
                "data_saver": "toggle_data_saver_mode", "roaming": "toggle_roaming", "data": "toggle_data"}
_CHECK_TOOL = {"airplane_mode": "check_status_bar", "wifi_calling": "check_wifi_calling_status",
               "wifi": "check_wifi_status", "data_saver": "check_data_restriction_status",
               "roaming": "check_network_status", "data": "check_network_status", "vpn": "check_vpn_status"}
_TOG_VERB = re.compile(r"\b(turn(?:ing|ed)?|switch(?:ing|ed)?|toggle|toggling|flip(?:ping)?|enable|enabling|disable|"
                       r"disabling|activate|deactivate|make sure|ensure|put|set)\b")
_CHK_VERB = re.compile(r"\b(check(?:ing)?|verify|confirm|see|look(?:ing)?|tell me|let me know|show|read|review|whether|"
                       r"find out|what|which|status of|is|are|do you see|are you seeing)\b")
_NEG = re.compile(r"\b(?:don'?t(?! worry| forget)|do not|no need|not need|need not|never|avoid|shouldn'?t|should not|"
                  r"won'?t|without|rather than|instead of|not necessary|unnecessary|no longer)\b")
_SELF = re.compile(r"\bi(?:'ll| will|'m going to| am going to|'ve| have| just| can| am)\b|\blet me (?!know)")  # agent describing its own step
_COND = re.compile(r"\b(?:if|otherwise|else|in case|unless|failing that|should (?:that|this|it) (?:not|fail))\b")
_POLITE_IF = re.compile(r"if you (?:don'?t mind|do not mind|could|can|would|wouldn'?t mind|have a (?:moment|minute|second)|"
                        r"are able|like|wish|want|please|are comfortable|are okay|are willing|'d like)\b|if possible|"
                        r"if you(?:'d| would) like")
_INDIRECT = re.compile(r"\b(?:check|checking|see|verify|confirm|determine|tell me|let me know|find out|know|ask|wonder)\s*$")  # "check if/whether ..." is a question, not a condition
_FAIL = re.compile(r"doesn'?t|does not|didn'?t|did not|isn'?t|not (?:work|help|fix|resolv)|still|fails?|failed|persist|"
                   r"no luck|unsuccessful|same|remain")
_FILLER = {"", "and", "but", "so", "now", "then", "okay", "ok", "well", "alright", "next"}


def _dir(verb: str, mid: str, post: str) -> Optional[bool]:
    """Direction of a toggle request from its verb, the text between verb and object, and the text after the object."""
    if verb.startswith(("enabl", "activat")):
        return True
    if verb.startswith(("disabl", "deactivat")):
        return False
    m = re.match(r"\s*(?:it\s+)?(?:back\s+|completely\s+|fully\s+)?(on|off)\b", mid)
    if m:
        return m.group(1) == "on"
    post = re.split(r"\b(?:and|then|but|or)\b|[,.;!?]", post)[0]
    p = re.match(r"\s*(?:back\s+)?(?:is\s+|are\s+|to\s+be\s+|should\s+be\s+|has\s+to\s+be\s+)?(?:now\s+|currently\s+|also\s+|still\s+)?"
                 r"(?:turned\s+|switched\s+|set\s+to\s+)?(on|off|enabled|disabled|active|inactive)\b", post)
    if p:
        return p.group(1) in ("on", "enabled", "active")
    if re.search(r"\b(?:off|disabled)\b", mid) or re.search(r"\b(?:not|isn'?t)\s+(?:in|on)\s*$", mid):
        return False
    return None


def _mode_from(s: str) -> Optional[str]:
    """Network-mode enum value named (or best implied) by `s` (tool signature: NetworkModePreference values)."""
    m = re.search(r"\b(4g_5g_preferred|4g_only|3g_only|2g_only)\b", s)
    if m:
        return m.group(1)
    if re.search(r"4g\s*(?:/|and|or|&|\+|,|-)\s*5g|5g\s*(?:/|and|or|&|\+|,|-)\s*4g|\b5g\b|4g\s*\+|up to 4g", s):
        return "4g_5g_preferred"
    if re.search(r"\b(?:4g|lte)\b", s):
        return "4g_only"
    if re.search(r"\b3g\b", s):
        return "3g_only"
    if re.search(r"\b2g\b", s):
        return "2g_only"
    return None


_PERM_SYN = (("storage", r"storage|files?|media|photos?|gallery"), ("sms", r"sms|text messag\w+|texting"),
             ("phone", r"phone"), ("network", r"network|internet access"))
_APP_SYN = (("messaging", r"messaging(?: app)?|messages app|messenger|text(?:ing)? app|sms app|message app"),
            ("browser", r"browser|chrome"))


def _app_in(s: str) -> Optional[str]:
    for name, pat in _APP_SYN:
        if re.search(rf"\b(?:{pat})\b", s):
            return name
    m = re.search(r"\b(?:for|to|of)\s+(?:the\s+|your\s+)?([a-z]+)\s+app\b", s)
    return m.group(1) if m else None


def _perms_in(s: str) -> list[str]:
    s = re.sub(r"\b(?:messaging|messages|sms|text(?:ing)?|message) app\b", " ", s)  # an app name is not a permission
    out: list[str] = []
    for name, pat in _PERM_SYN:
        if re.search(rf"\b(?:{pat})\b", s) and name not in out:
            out.append(name)
    return out


def _da(tool: str, ensure: Optional[tuple] = None, **args: Any) -> DeviceAction:
    return DeviceAction("user", tool, _frozen_items({k: v for k, v in args.items() if v is not None}), ensure)


class RuleActionParser:
    """Lexicon/regex parser over all 30 phone tools (see module docstring for the conditional/negation rules)."""

    # ---- sentence-level machinery
    @staticmethod
    def _cond_info(sent: str) -> tuple[Optional[int], Optional[tuple[int, int]]]:
        """(conditional-from offset or None, condition-clause span whose matches are dropped or None)."""
        for m in _COND.finditer(sent):
            if _POLITE_IF.match(sent, m.start()) or _INDIRECT.search(sent[max(0, m.start() - 20):m.start()]):
                continue
            p, w = m.start(), m.group(0)
            if w in ("otherwise", "else", "failing that"):
                return p, None
            c = re.search(r",|\bthen\b", sent[m.end():])
            cend = m.end() + c.start() if c else len(sent)
            initial = sent[:p].strip(" ,") in _FILLER
            if initial:
                return 0, (p, cend)
            if cend == len(sent):  # trailing condition: "restart it, if that doesn't help"
                return (0 if _FAIL.search(sent[p:]) else None), (p, cend)
            return p, (p, cend)
        return None, None

    def _sentence_actions(self, sent: str, prior_obj: Optional[str], ctx: dict) -> tuple[list[tuple[int, DeviceAction]], Optional[str]]:
        out: list[tuple[int, DeviceAction]] = []
        used: list[tuple[int, int]] = []
        last_obj = prior_obj

        def free(a: int, b: int) -> bool:
            return not any(a < y and x < b for x, y in used)

        def add(a: int, b: int, act: DeviceAction, neg_from: Optional[int] = None) -> None:
            if not free(a, b):
                return
            pre = sent[:a]
            pre = re.split(r"[,:—]", pre)[-1] if neg_from is None else sent[neg_from:a]
            if _NEG.search(pre) or _SELF.search(pre):
                used.append((a, b))  # consumed, but negated / the agent's own action: not a request to the user
                return
            used.append((a, b))
            out.append((a, act))

        def cue_before(pos: int, pat: str) -> bool:
            return bool(re.search(pat, sent[:pos]))

        REQ = (r"\b(?:check|verify|confirm|see|look|tell me|let me know|show|read|run|test|try|open|go to|please|can you|"
               r"could you|would you|let'?s|first|next|now|find out|what|which|whether|need|want|take|do)\b")

        # ---- reboot
        for m in re.finditer(r"\b(?:restart|reboot)\w*\s+(?:your|the)\s+(?:phone|device|handset|mobile)\b|\b(?:restart|reboot)\s+it\b|"
                             r"(?<!a )(?<!the )(?<!after )(?<!before )(?<!until )(?<!once )(?<!for )(?<!next )(?<!upon )\breboot_device\b|"
                             r"(?<!a )(?<!the )(?<!after )(?<!before )(?<!until )(?<!once )(?<!for )(?<!next )(?<!upon )\breboot\b(?!\w)|"
                             r"\bpower[- ]?cycl\w+|\bturn(?:ing)? (?:your phone|the phone|your device|the device|it) off and (?:back )?on\b|"
                             r"\bshut (?:it|your phone|the phone) (?:down )?and (?:back )?(?:on|up)\b|\brestart(?:ing)? (?:the )?device\b", sent):
            add(m.start(), m.end(), _da("reboot_device"))
        # ---- SIM
        for m in re.finditer(r"\breseat\w*|\breinsert\w*\s+(?:the |your )?(?:sim|card)|"
                             r"\b(?:remove|take out|pull out|eject|unplug)\w*\s+(?:the |your )?(?:sim\b)?[\w\s]{0,12}?(?:and|then|&)\s+(?:put|place|insert|replace|reinsert)\w*|"
                             r"\b(?:take|pull|remove|eject)\w*\s+(?:out\s+)?(?:the |your )?sim\b|\breseat_sim_card\b", sent):
            add(m.start(), m.end(), _da("reseat_sim_card"))
        for m in re.finditer(r"\bsim(?: card)?(?: status| state)\b|\bcheck_sim_status\b|"
                             r"(?:check|verify|confirm|see|look|whether|tell me|status of|inspect)\w*[^.?!]{0,25}\bsim\b|"
                             r"\bis (?:your |the )?sim(?: card)? (?:active|working|recognized|detected|inserted|locked|ok)", sent):
            add(m.start(), m.end(), _da("check_sim_status"))
        # ---- network mode
        for m in re.finditer(r"\b(?:set\w*|chang\w*|switch\w*|updat\w*|select\w*|choos\w*|adjust\w*|us(?:e|ing))\s+(?:your\s+|the\s+|phone'?s\s+)*"
                             r"(?:preferred\s+)?network\s+(?:mode|type|preference)s?\b([^.;!?]{0,60})|"
                             r"\bset_network_mode_preference\b([^.;!?]{0,60})|"
                             r"\b(?:switch\w*|chang\w*|set\w*|mov\w*)\s+(?:it |your phone |your network |to )?(?:over )?to\s+(?:a |the )?"
                             r"(?:network mode (?:that )?(?:includes|with)?[^.;!?]{0,20}|(?:4g|5g|lte|3g|2g)\b[^.;!?]{0,20})", sent):
            tail = sent[m.start():m.end() + 40]
            add(m.start(), m.end(), _da("set_network_mode_preference", mode=_mode_from(tail)))
        for m in re.finditer(r"\bnetwork (?:mode|type)(?: preference| setting)?s?\b|\bpreferred network\b|\bnetwork preference\b|"
                             r"\bcheck_network_mode_preference\b", sent):
            if free(m.start(), m.end()) and cue_before(m.end(), REQ):
                add(m.start(), m.end(), _da("check_network_mode_preference"))
        # ---- APN
        for m in re.finditer(r"\bapn\b|\breset_apn_settings\b|\bset_apn_settings\b|\bcheck_apn_settings\b", sent):
            if not free(m.start(), m.end()):
                continue
            pre = sent[max(0, sent.rfind(",", 0, m.start()) + 1):m.start()]
            post = sent[m.end():m.end() + 40]
            if "reset_apn" in m.group(0) or re.search(r"\b(?:reset|restore|default|revert|clear)\w*\b", pre + " " + post[:20]):
                add(m.start(), m.end(), _da("reset_apn_settings"))
            elif "set_apn" in m.group(0) or re.search(r"\b(?:set\w*|chang\w*|updat\w*|configur\w*|correct\w*|fix\w*)\b", pre):
                add(m.start(), m.end(), _da("set_apn_settings", apn_settings=(_frozen_items({"apn_name": "internet"}) if "internet" in post else None)))
            elif "check_apn" in m.group(0) or cue_before(m.end(), REQ):
                add(m.start(), m.end(), _da("check_apn_settings"))
        # ---- permissions
        for m in re.finditer(r"\bgrant_app_permission\b|\b(?:grant|give|allow|enable)\w*\b[^.;!?]{0,60}?\b(?:permissions?|access)\b[^.;!?]{0,60}", sent):
            if not free(m.start(), m.end()) or re.search(r"\bcheck\w*\b", sent[m.start():m.start() + 12]):
                continue
            app = _app_in(sent) or "messaging"  # grant is only used for the messaging app in this domain
            perms = _perms_in(sent[m.start():m.end() + 40]) or [None]
            pre = re.split(r"[,:\u2014]", sent[:m.start()])[-1]
            used.append((m.start(), m.end()))
            if _NEG.search(pre):
                continue
            for p in perms:
                out.append((m.start(), _da("grant_app_permission", app_name=app, permission=p)))
        if re.search(r"\bpermissions?\b", sent):
            for m in re.finditer(r"\b(?:enable|allow|turn on|switch on|toggle on)\s+(?:both\s+|the\s+)?((?:storage|sms|phone|network)\b(?:\s*(?:,|and|&)\s*(?:storage|sms|phone|network)\b)*)", sent):
                if free(m.start(), m.end()):
                    used.append((m.start(), m.end()))
                    for pm in _perms_in(m.group(1)):
                        out.append((m.start(), _da("grant_app_permission", app_name=_app_in(sent) or "messaging", permission=pm)))
        for m in re.finditer(r"\bcheck_app_permissions\b|\bpermissions?\b|\bcheck_app_status\b|\bapp status\b|"
                             r"\bbackground data\b|\bstatus of (?:the |your )?[a-z]+ app\b", sent):
            if not free(m.start(), m.end()) or not cue_before(m.end(), REQ) or any(a.tool == "grant_app_permission" for _, a in out):
                continue
            app = _app_in(sent) or "messaging"
            tool = "check_app_status" if re.search(r"status|background data", m.group(0)) else "check_app_permissions"
            add(m.start(), m.end(), _da(tool, app_name=app))
        for m in re.finditer(r"\binstalled apps?\b|\bapps (?:are |that are |you have )?installed\b|\blist of (?:installed )?apps\b|"
                             r"\bwhich apps\b|\bcheck_installed_apps\b", sent):
            if cue_before(m.end(), REQ) or "check_installed" in m.group(0):
                add(m.start(), m.end(), _da("check_installed_apps"))
        # ---- speed test / MMS / payment
        for m in re.finditer(r"\bspeed ?test\b|\brun_speed_test\b|\b(?:test|measure|check|see|run)\w*\s+(?:your\s+|the\s+|how fast\s+)?"
                             r"(?:internet |data |download |connection |network )?speed\b|\bhow fast\b", sent):
            if cue_before(m.end(), REQ) or m.group(0) == "run_speed_test" or re.match(r"(?:test|measure|check|run)", m.group(0)):
                add(m.start(), m.end(), _da("run_speed_test"))
        for m in re.finditer(r"\bcan_send_mms\b|\b(?:send|sending|sent)\s+(?:a |an |the |some )?(?:test )?(?:picture|photo|image|multimedia|pic|mms)\w*|"
                             r"\b(?:test|check|try|verify)\w*\s+(?:sending\s+)?(?:an? |the )?mms\b|\bmms\s+(?:test|capability)\b", sent):
            if re.search(r"\b(?:send|sending|try|test|check|verify|see|able|can you|could you|attempt|please)\b", sent[:m.end()] + " send"):
                add(m.start(), m.end(), _da("can_send_mms"))
        for m in re.finditer(r"\b(?:make|submit|complete|approve|accept|authori[sz]e|process)\w*\s+(?:the\s+|that\s+|this\s+|your\s+)?"
                             r"(?:payment|bill)\b|\bpay\s+(?:the|that|this|your)\s+(?:bill|request|invoice|amount|balance|payment)\b|"
                             r"\bgo ahead and pay\b|\bmake_payment\b|\b(?:approve|accept)\s+(?:the |that |this )?(?:payment )?request\b", sent):
            if re.search(r"\b(?:please|you|can you|could you|go ahead|approve|accept|pay)\b", sent[:m.end()]) or "make_payment" in m.group(0):
                add(m.start(), m.end(), _da("make_payment"))
        for m in re.finditer(r"\b(?:accept|approve|pay|authori[sz]e)\s+(?:it|that|this|them)\b", sent):
            if ctx.get("pay") and free(m.start(), m.end()):
                add(m.start(), m.end(), _da("make_payment"))
        if re.search(r"\bpayment request|\bbill\b", sent):
            ctx["pay"] = True
        for m in re.finditer(r"\bpayment requests?\b|\bcheck_payment_request\b", sent):
            if free(m.start(), m.end()) and (cue_before(m.end(), r"\b(?:check|see|received|receive|got|look|any|seen|find|open)\b")
                                             or "check_payment" in m.group(0)):
                add(m.start(), m.end(), _da("check_payment_request"))
        # ---- status bar / network status
        for m in re.finditer(r"\bstatus bar\b|\bcheck_status_bar\b|\b(?:icons|symbols|indicators)\b[^.;!?]{0,30}\b(?:top|screen|status)\b|"
                             r"\btop of (?:your |the )?screen\b|\bsignal (?:bars|icon|indicator)\b|\bhow many bars\b|"
                             r"\bwhat (?:icons|symbols|do you see)\b|\bicons? (?:do you|are|you)\b", sent):
            if free(m.start(), m.end()) and (cue_before(m.end(), REQ) or "check_status_bar" in m.group(0) or re.match(r"what|how|icon", m.group(0))):
                add(m.start(), m.end(), _da("check_status_bar"))
        for m in re.finditer(r"\bnetwork (?:status|info|information|connection status)\b|\bcellular (?:status|connection)\b|"
                             r"\bconnection status\b|\bcheck_network_status\b|\bnetwork settings\b", sent):
            if free(m.start(), m.end()) and (cue_before(m.end(), REQ) or "check_network_status" in m.group(0)):
                add(m.start(), m.end(), _da("check_network_status"))
        # ---- toggles / checks on named settings (airplane, wifi, wifi calling, data, roaming, data saver, vpn)
        for m in _OBJ.finditer(sent):
            kind = m.lastgroup
            if not free(m.start(), m.end()):
                continue
            last_obj = kind
            clause_start = max(sent.rfind(",", 0, m.start()), sent.rfind(":", 0, m.start()), sent.rfind("—", 0, m.start())) + 1
            pre = sent[clause_start:m.start()]
            post = sent[m.end():m.end() + 30]
            tv = None
            for tv in _TOG_VERB.finditer(pre):
                pass
            cv = None
            for cv in _CHK_VERB.finditer(pre):
                pass
            if kind == "vpn":
                self._vpn(sent, m, pre, post, add, cv)
                continue
            if tv is not None and (cv is None or tv.start() >= cv.start() or tv.group(0) in ("make sure", "ensure")):
                mid = pre[tv.end():]
                if len(mid.strip()) <= 45 and not re.search(r"\b(?:check|see|verify|confirm|look|tell)\b", mid):
                    d = _dir(tv.group(0), mid, post)
                    add(m.start(), m.end(), _da(_TOGGLE_TOOL[kind], ensure=(kind, d) if d is not None else None), clause_start)
                    continue
            if (cv is not None or re.search(r"\b(?:is|are)\s*$", pre)) and not re.match(r"\s*(?:was|were)\b", post):
                if cv is not None or sent.rstrip().endswith("?"):
                    add(m.start(), m.end(), _da(_CHECK_TOOL[kind]), clause_start)
                    continue
            if re.match(r"\s*(?:status|settings)\b", post) and kind != "airplane_mode":
                add(m.start(), m.end(), _da(_CHECK_TOOL[kind]))
            elif re.match(r"\s*(?:needs? to|should|must|has to|will need to)\s+be\s+(?:turned\s+|switched\s+)?(on|off|enabled|disabled)\b", post):
                d = _dir("", "", re.sub(r"^\s*(?:needs? to|should|must|has to|will need to)\s+be", "", post))
                add(m.start(), m.end(), _da(_TOGGLE_TOOL[kind], ensure=(kind, d) if d is not None else None))
        # ---- pronoun toggles: "turn it back off" refers to the previous setting
        for m in re.finditer(r"\b(?:turn|switch|flip|toggle)\s+(?:it|that|this)\s+(?:back\s+)?(on|off)\b", sent):
            if free(m.start(), m.end()) and last_obj is not None:
                d = m.group(1) == "on"
                if last_obj == "vpn":
                    add(m.start(), m.end(), _da("connect_vpn" if d else "disconnect_vpn", ensure=("vpn", d)))
                else:
                    add(m.start(), m.end(), _da(_TOGGLE_TOOL[last_obj], ensure=(last_obj, d)))
        # ---- literal tool names (agents sometimes write `toggle_wifi()`)
        for name in USER_TOOL_NAMES:
            for m in re.finditer(rf"\b{name}\b", sent):
                if not free(m.start(), m.end()):
                    continue
                post = sent[m.end():m.end() + 45]
                args: dict[str, Any] = {}
                ens = None
                if name.startswith("toggle_") or name in ("connect_vpn", "disconnect_vpn"):
                    key = {v: k for k, v in _TOGGLE_TOOL.items()}.get(name, "vpn")
                    d = (name == "connect_vpn") if key == "vpn" and name != "toggle_vpn" else None
                    if d is None:
                        t = re.search(r"\b(?:turn|switch)\w*\s+(?:it\s+)?(?:back\s+)?(on|off)\b|\bto\s+(enable|disable)\b", post)
                        if t:
                            d = (t.group(1) or ("on" if t.group(2) == "enable" else "off")) == "on"
                    ens = (key, d) if d is not None else None
                if name == "set_network_mode_preference":
                    args["mode"] = _mode_from(post)
                if name == "grant_app_permission":
                    args["app_name"] = _app_in(post) or "messaging"
                    ps = _perms_in(post)
                    args["permission"] = ps[0] if ps else None
                if name in ("check_app_status", "check_app_permissions"):
                    args["app_name"] = _app_in(post) or "messaging"
                add(m.start(), m.end(), _da(name, ens, **args))
        return sorted(out, key=lambda x: x[0]), last_obj

    def _vpn(self, sent, m, pre, post, add, cv) -> None:
        verbs = list(re.finditer(r"\b(disconnect\w*|reconnect\w*|connect\w*|turn(?:ing)? off|switch(?:ing)? off|turn(?:ing)? on|"
                                 r"switch(?:ing)? on|disable|enable|stop|start|shut off|activate|deactivate)\b", pre))
        if verbs:
            v = verbs[-1].group(0)
            off = v.startswith(("disconnect", "disable", "deactivate", "stop", "shut")) or v.endswith("off")
            add(verbs[-1].start() + (len(sent[:m.start()]) - len(pre)), m.end(),
                _da("disconnect_vpn" if off else "connect_vpn", ensure=("vpn", not off)))
        elif cv is not None:
            add(len(sent[:m.start()]) - len(pre) + cv.start(), m.end(), _da("check_vpn_status"))
        elif re.match(r"\s*(?:status|connection)\b", post):
            add(m.start(), m.end(), _da("check_vpn_status"))

    # ---- public API
    def parse_all(self, agent_text: str) -> tuple[ParsedAction, ...]:
        """Every phone request in textual order, each flagged `conditional` (see module docstring)."""
        t = _norm(agent_text)
        res: list[ParsedAction] = []
        seen: set = set()
        last_obj: Optional[str] = None
        ctx: dict = {}  # message-level context: a payment/bill was mentioned (so "accept it" is make_payment)
        for a, b in _sentence_spans(t):
            sent = t[a:b]
            cond_from, drop = self._cond_info(sent)
            found, last_obj = self._sentence_actions(sent, last_obj, ctx)
            for pos, act in found:
                if drop is not None and drop[0] <= pos < drop[1]:
                    continue
                key = (act.tool, act.args, act.ensure)
                if key in seen:
                    continue
                seen.add(key)
                res.append(ParsedAction(act, cond_from is not None and pos >= cond_from, a + pos, sent))
        return tuple(sorted(res, key=lambda r: r.start))

    def parse(self, agent_text: str) -> tuple[DeviceAction, ...]:
        allp = self.parse_all(agent_text)
        firm = tuple(p.action for p in allp if not p.conditional)
        return firm or tuple(p.action for p in allp[:1])

    def classify_agent_message(self, agent_text: str, claim: Optional["RuleClaimClassifier"] = None) -> DecisionKind:
        """Decision kind of a text-only agent message (backend tool calls are structured and need no parsing)."""
        return classify_agent_message(agent_text, self, claim)


_TRANSFER_RE = re.compile(r"\b(?:transfer(?:ring)? you|(?:i(?:'ll| will| am going to|'m going to)|let me) (?:now )?(?:transfer|escalate|connect you)|"
                          r"escalat\w+ (?:this|your|the)|connect(?:ing)? you (?:to|with) (?:a |an |our )?(?:human|live|technical|specialist|support|senior)|"
                          r"hand(?:ing)? you (?:over )?to (?:a |an |our )?(?:human|specialist|technical))")


def classify_agent_message(text: str, actions: Optional[RuleActionParser] = None,
                           claim: Optional["RuleClaimClassifier"] = None) -> DecisionKind:
    """TRANSFER-text > CLAIM_RESOLVED (claim prob >= 0.5 and no firm device request) > REQUEST_DEVICE > ASK."""
    actions = actions or RuleActionParser()
    claim = claim or RuleClaimClassifier()
    t = _norm(text)
    if _TRANSFER_RE.search(t):
        return DecisionKind.TRANSFER
    allp = actions.parse_all(text)
    firm = [p for p in allp if not p.conditional]
    if claim.prob_claim(text) >= 0.5 and not firm:
        return DecisionKind.CLAIM_RESOLVED
    if allp:
        return DecisionKind.REQUEST_DEVICE
    return DecisionKind.ASK


# =========================================================================== claim classifier
_CLAIM_STRONG = (
    (re.compile(r"\b(?:issue|problem|it|everything|that|this|service|connection|phone|mms|data)\b[^.?!]{0,30}\b(?:is|has been|have been|should be|should now be|is now|are now|appears to be|seems to be|are)\b[^.?!]{0,15}\b(?:resolved|fixed|working|sorted|restored|solved|back to normal|good to go|all set|up and running)\b"), 0.95),
    (re.compile(r"\b(?:i'?ve|i have|we'?ve|we have)\b[^.?!]{0,25}\b(?:resolved|fixed|solved|restored|sorted)\b"), 0.95),
    (re.compile(r"\b(?:glad|happy|pleased) (?:i|we) (?:could|was able to|were able to|managed to)"), 0.92),
    (re.compile(r"\b(?:is|are) (?:all )?(?:set|good|working fine|working properly)\b|\bback (?:up|online|to normal)\b"), 0.8),
    (re.compile(r"\bis there anything else\b|\banything else (?:i|we) can (?:help|assist|do)\b|\banything else you need\b|\bif you need (?:anything|further|any more|more)\b|\bfeel free to (?:reach out|contact)\b"), 0.85),
    (re.compile(r"\byou (?:should|can|will) (?:now )?be able to (?:send|use|browse|make|connect|access)\b"), 0.8),
    (re.compile(r"\b(?:should|will) (?:now )?(?:be )?(?:working|fixed|resolved|back)\b"), 0.75),
    (re.compile(r"\bthat (?:should|will) (?:fix|resolve|solve|do it|take care|sort)\b"), 0.6),
)
_CLAIM_VERIFY = re.compile(r"\b(?:let'?s|let me|please|can you|could you|would you)\b[^.?!]{0,40}\b(?:verify|confirm|check|test|run|try|see if|let me know)\b|"
                           r"\bto (?:verify|confirm|make sure|test)\b|\bis it (?:working|fixed)\b|\bdoes (?:it|that) (?:work|help)\b|"
                           r"\bare you (?:now )?able to\b|\b(?:can|could) you now\b|\bonce you(?:'ve| have)?\b|\bafter (?:that|you|the reboot|restarting)\b|"
                           r"\bif (?:it|that|this) (?:still|doesn'?t|does not|isn'?t)\b|\bnext(?:,)? (?:i|let'?s|we)\b|\bwe(?:'ll| will| can) (?:try|check|need)\b|\bhelp us\b")
_CLAIM_HYPO = re.compile(r"(?<!can )(?<!to )(?<!i'll )\b(?:confirm|verify|check|see (?:if|whether)|whether|make sure|determine|test)\b")
_CLAIM_CLOSURE = re.compile(r"anything else|glad (?:i|we)|feel free to|happy to help")


class RuleClaimClassifier:
    """P(an agent message claims the issue is resolved). Pattern evidence with a verification-request penalty."""

    def prob_claim(self, agent_text: str) -> float:
        t = _norm(agent_text)
        if not t:
            return 0.01
        view = " . ".join(_CLAIM_HYPO.split(x)[0] for x in re.split(r"(?<=[.!?])\s+|\n+", t))  # "so we can confirm it is fixed" is not a claim
        pos = max((w for rx, w in _CLAIM_STRONG if rx.search(view)), default=0.0)
        ver = bool(_CLAIM_VERIFY.search(t))
        if pos == 0.0:
            return 0.01 if ver else 0.03
        if ver:
            pos *= 0.9 if _CLAIM_CLOSURE.search(t) else (0.6 if pos >= 0.9 else 0.3)
        if t.rstrip().endswith("?") and not _CLAIM_CLOSURE.search(t):
            pos *= 0.5
        return float(min(0.99, max(0.01, pos)))


# =========================================================================== intent parser
_INTENT_CUES: dict[str, tuple[re.Pattern, ...]] = {
    "service_issue": (re.compile(r"\bno (?:service|signal|bars|reception|cell(?:ular)? (?:service|signal))\b|\bno_service\b|\bnot connected to (?:the |any )?(?:cellular |mobile )?network\b|"
                                 r"\bsearching for (?:network|service|signal)\b|\bemergency calls only\b|\blost (?:service|signal|reception)\b|\bcan'?t (?:make|place|receive) (?:any )?(?:phone )?calls?\b|"
                                 r"\bcannot (?:make|place|receive) (?:any )?(?:phone )?calls?\b|\bno connection to (?:the )?network\b|\bnot getting (?:any )?(?:service|signal|bars)\b|\bzero bars\b|\bphone (?:has|shows|says) no\b"),),
    "mobile_data_issue": (re.compile(r"\b(?:mobile |cellular )?data\b[^.?!]{0,25}\b(?:slow|not working|isn'?t working|doesn'?t work|won'?t work|stopped|keeps? (?:dropping|cutting)|too slow|very slow|bad|poor)\b|"
                                     r"\b(?:slow|poor|bad|terrible|very slow)\s+(?:mobile |cellular )?(?:internet|data|speeds?|connection|browsing|downloads?)\b|"
                                     r"\bcan'?t (?:browse|load|get online|use the internet|access the internet|stream|open websites)\b|\binternet\b[^.?!]{0,20}\b(?:slow|not working|down|isn'?t working)\b|"
                                     r"\bno (?:internet|data)\b|\b(?:web)?pages? (?:won'?t|don'?t|take) (?:load|forever)\b|\bapps? (?:won'?t|don'?t) load\b|\bbuffering\b|\bspeeds? (?:are|is) (?:slow|terrible|bad)\b|\bdata (?:is )?(?:very )?slow\b|"
                                     r"\b(?:trouble|issues?|problems?)\s+with\s+(?:my\s+)?(?:mobile|cellular)\s+data\b|\b(?:not working|stops? working)\s+or\s+(?:is\s+)?(?:very\s+|really\s+)?slow\b"),),
    "mms_issue": (re.compile(r"\bmms\b|\bpicture messag\w+|\bphoto messag\w+|\bmultimedia\b|\b(?:can'?t|cannot|unable to|not able to|won'?t|couldn'?t)\b[^.?!]{0,25}\b(?:send|sending|receive)\b[^.?!]{0,25}\b(?:pictures?|photos?|images?|pics?|attachments?|media)\b|"
                           r"\b(?:pictures?|photos?|images?|pics?)\b[^.?!]{0,25}\b(?:won'?t|will not|can'?t|fail\w*|not)\b[^.?!]{0,10}\b(?:send|sending|go through|attach)\b|\bgroup (?:text|message)s?\b[^.?!]{0,25}\b(?:pictures?|photos?)\b"),),
}


class RuleIntentParser:
    """First user message -> intent probabilities. Cue counts smoothed with alpha, floored, normalised to sum 1."""

    alpha = 0.1
    floor = 0.02

    def parse(self, first_user_text: str) -> dict[str, float]:
        t = _norm(first_user_text)
        hits = {i: float(sum(1 for rx in _INTENT_CUES[i] for _ in rx.finditer(t))) for i in INTENTS}
        # one cue family hit counts once per distinct match, but is capped so a rambling message cannot dominate
        hits = {i: min(h, 3.0) for i, h in hits.items()}
        tot = sum(hits.values())
        if tot == 0:
            return {i: 1.0 / len(INTENTS) for i in INTENTS}
        raw = {i: (hits[i] + self.alpha) / (tot + self.alpha * len(INTENTS)) for i in INTENTS}
        raw = {i: max(p, self.floor) for i, p in raw.items()}
        z = sum(raw.values())
        return {i: p / z for i, p in raw.items()}


# =========================================================================== observation parser
@dataclass(frozen=True)
class ParsedObservation(Observation):
    """An `Observation` that also records what the parser refused to emit (the contract is unchanged)."""
    conflicts: tuple = ()  # ((field, (value, ...)), ...): self-contradicting present-tense statements, not emitted
    out_of_space: tuple = ()  # ((field, value), ...): extracted but outside the closed label space, not emitted


_OFF_W = r"(?:off|disabled|deactivated|gone|missing|hidden|absent|disappeared|inactive|not (?:showing|visible|there|displayed|shown|present|on|enabled|active))"
_ON_RE = re.compile(r"\b(?:on(?!\s+(?:the|my|your|a|an|this|that|it|top|screen|phone|status|home|and\s+off)\b)|enabled|activated|active|showing|visible|shown|displayed|there|present|appears?|appearing|lit|connected|yes)\b")
_OFF_RE = re.compile(r"\b(?:off|disabled|deactivated|gone|missing|hidden|absent|disappeared|inactive|no)\b")
_NEGON_RE = re.compile(r"(?:\bnot\b|n't\b|\bno longer\b|\bnever\b)\W+(?:\w+\W+){0,1}?(?:on|enabled|active|showing|visible|shown|displayed|there|present|connected)\b")
_PAST_RE = re.compile(r"\b(?:was|were)\s+(?:still\s+|already\s+)?(?:on|off|enabled|disabled|active|connected|showing|there|missing|set|slow|fine)\b|"
                      r"\bused to\b|\bearlier\b|\bpreviously\b|\binitially\b|\bat first\b|\bbefore (?:i|the|that)\b|\bhad been\b|\ba while ago\b|\byesterday\b")
_NUMS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _bool_state(c: str, obj: re.Pattern, *, yesno: bool = False) -> list[bool]:
    """Every True/False state statement about the object in `c` (True = on/shown/enabled, False = off/gone/disabled)."""
    out: list[bool] = []
    for m in obj.finditer(c):
        pre = c[max(0, m.start() - 28):m.start()]
        pre = re.split(r"[,;:]", pre)[-1]
        post = re.split(r"[.;!?,]|\bbut\b|\band\b(?!\s+(?:now\s+)?(?:it|that)\b)|\bso\b|\bthen\b", c[m.end():m.end() + 40])[0]
        v: Optional[bool] = None
        neg_pre = re.search(r"(?:\bno\b|\bnot\b|n't\b|\bwithout\b|\bnever\b)(?:\s+\w+){0,3}\s*$", pre)
        yn = re.match(r"\s*(?:enabled|radio|connected)?\s*:?\s*(yes|no)\b", post) if yesno else None
        if yn:
            v = yn.group(1) == "yes"
        elif _NEGON_RE.search(post):
            v = False
        elif re.search(rf"\b{_OFF_W}\b", post) and not re.search(r"\bturn\w*\s+(?:it\s+)?on\b", post):
            v = False
        elif _ON_RE.search(post):
            v = True
        elif re.search(r"\b(?:turn\w*|switch\w*|flip\w*)\s+(?:it\s+)?(?:back\s+)?off\b|\bdisabl\w+|\bdeactivat\w+", pre):
            v = False
        elif re.search(r"\b(?:turn\w*|switch\w*|flip\w*)\s+(?:it\s+)?(?:back\s+)?on\b|\benabl\w+|\bactivat\w+", pre):
            v = True
        elif neg_pre:
            v = False
        elif re.search(r"\b(?:see|seeing|there'?s|there is|shows?|showing|displays?|have|has|got|notice)\b[^.]{0,15}$", pre):
            v = True
        if neg_pre and v is True:
            v = False
        if v is not None:
            out.append(v)
    return out


_RX = {
    "airplane": re.compile(r"airplane(?: mode| icon| symbol| indicator)?|plane (?:icon|symbol)"),
    "wifi_calling": re.compile(r"wifi calling|wifi calls?"),
    "wifi": re.compile(r"wifi(?: radio)?(?! calling)(?! signal)"),
    "data_saver": re.compile(r"data sav(?:er|ing)(?: mode| icon)?"),
    "roaming": re.compile(r"(?:data )?roaming"),
    "data": re.compile(r"(?:mobile|cellular) data|data connection|data icon|\bdata\b(?!\s+(?:saver|saving|roaming|usage|plan|limit|used|allowance))"),
    "vpn": re.compile(r"\bvpn\b"),
}
_PERMS = ("sms", "storage", "phone", "network")


class RuleObservationParser:
    """User text -> `ParsedObservation` over a closed label space (see module docstring)."""

    # ---- per-clause fact extraction; each fact -> list of values
    def _facts(self, text: str, requested: Sequence[DeviceAction]) -> dict[str, list]:
        facts: dict[str, list] = {}

        def put(k: str, v: Any) -> None:
            facts.setdefault(k, []).append(v)

        subject: Optional[str] = None  # last setting named anywhere (even in a past clause), for "I turned it off"
        key_of = {"airplane": "airplane", "wifi": "wifi_on", "wifi_calling": "wifi_calling", "data": "data",
                  "data_saver": "data_saver", "roaming": "roaming", "vpn": "vpn"}
        for a, b in _sentence_spans(text):
            segs = [x for x in re.split(r"\s*(?:,|;|\bbut\b|\bso\b|\bthen\b)\s*", text[a:b]) if x]
            keep = []
            for seg in segs:
                named = [(m.start(), k) for k, rx in _RX.items() for m in rx.finditer(seg)]
                pm = re.search(r"\b(?:turn|switch|flip)\w*\s+(?:it|that)\s+(?:back\s+)?(on|off)\b", seg)
                if pm and not named and subject in key_of:
                    put(key_of[subject], pm.group(1) == "on")
                if named:
                    subject = max(named)[1]
                if not _PAST_RE.search(seg):
                    keep.append(seg)
            if keep:
                self._clause(", ".join(keep), put, requested)
        return facts

    def _clause(self, c: str, put, requested: Sequence[DeviceAction]) -> None:
        # airplane mode
        for v in _bool_state(c, _RX["airplane"]):
            put("airplane", v)
        if re.search(r"\bonly\b[^.]{0,15}airplane|airplane[^.]{0,20}(?:\bonly\b|nothing else|and nothing)", c):
            for k in ("signal_hidden",):
                put(k, True)
        # signal / bars
        if not re.search(r"wifi[^.]{0,12}(?:signal|bars?)", c):
            sig = None
            if re.search(r"\b(?:no|zero|0)\s+(?:signal|bars?|service|reception)\b|no_?service|no_signal|signal\W{0,4}none\b|not getting any (?:signal|service|bars)|"
                         r"(?:can'?t|cannot) get (?:any )?(?:signal|service)|\bnone\b[^.]{0,8}\bsignal\b|὏5 no signal", c):
                sig = "none"
            elif re.search(r"\b(?:signal|bars?|reception)\b|\U0001F4F6", c):
                n = re.search(r"\b(\d|one|two|three|four|five)\s*(?:out of \w+\s+)?(?:signal )?bars?\b", c)
                num = (int(n.group(1)) if n.group(1).isdigit() else _NUMS[n.group(1)]) if n else None
                if num is not None:
                    sig = {0: "none", 1: "poor", 2: "fair", 3: "good"}.get(num, "excellent")
                elif re.search(r"\b(?:poor|weak|low|very weak|barely)\b", c):
                    sig = "poor"
                elif re.search(r"\b(?:fair|medium|moderate|so-so)\b", c):
                    sig = "fair"
                elif re.search(r"\b(?:excellent|full|strong|great|all bars)\b", c):
                    sig = "excellent"
                elif re.search(r"\b(?:good|decent|solid)\b", c):
                    sig = "good"
            if sig:
                put("signal", sig)
        if re.search(r"\bno_?service\b|\bno service\b", c):
            put("conn", "no_service")
        elif re.search(r"\bsearching\b", c):
            put("conn", "searching")
        elif re.search(r"emergency (?:calls )?only|\bsos only\b|emergency_only", c):
            put("conn", "emergency_only")
        elif re.search(r"cellular connection\W+connected|connected to (?:the |my )?(?:carrier|cellular network|mobile network)", c):
            put("conn", "connected")
        # network mode preference vs. network technology
        mode = None
        if re.search(r"\bprefer\w*|\bmode\b|set to|_only|_preferred|\bonly\b", c):
            if re.search(r"4g_5g_preferred|4g\s*(?:/|and|or|&|\+|,)\s*5g|5g\s*(?:/|and|or|&)\s*4g|5g preferred", c):
                mode = "4g_5g_preferred"
            elif re.search(r"\b(?:4g|lte)\b[\s_]*only|4g_only", c):
                mode = "4g_only"
            elif re.search(r"\b3g\b[\s_]*only|3g_only", c):
                mode = "3g_only"
            elif re.search(r"\b2g\b[\s_]*only|2g_only", c):
                mode = "2g_only"
        if mode:
            put("mode", mode)
            if re.search(r"invalid|not a valid", c):
                put("mode_invalid", True)
        else:
            if re.search(r"network type\W{1,3}none|no network type", c):
                put("tech", "none")
            techs = {"2g": "2G", "3g": "3G", "4g": "4G", "5g": "5G", "lte": "4G"}
            for tm in re.finditer(r"\b([2-5]g|lte)\b", c):
                put("tech", techs[tm.group(1)])
        # mobile data / saver / roaming / wifi / wifi calling / vpn
        for v in _bool_state(c, _RX["data"], yesno=True):
            if not re.search(r"data\s+(?:is\s+)?(?:slow|not working)", c) or v is not None:
                put("data", v)
        for v in _bool_state(c, _RX["data_saver"]):
            put("data_saver", v)
        for v in _bool_state(c, _RX["roaming"], yesno=True):
            put("roaming", v)
        for v in _bool_state(c, _RX["wifi_calling"]):
            put("wifi_calling", v)
        for v in _bool_state(c, _RX["wifi"], yesno=True):
            put("wifi_on", v)
        if re.search(r"(?:not|isn'?t|aren'?t|no longer|never)\s+(?:currently\s+)?connected to (?:any |the )?(?:a )?(?:wifi|network|anything)|no wifi|don'?t have (?:any )?wifi|wifi (?:is |isn'?t )?not connected|wifi connected\W+no", c):
            put("wifi_conn", False)
        elif re.search(r"connected to (?:the |my |a )?(?:wifi|home wifi|\w+ wifi)|wifi (?:is )?connected|wifi connected\W+yes", c):
            put("wifi_conn", True)
        self._vpn(c, put)
        # SIM
        if re.search(r"\bsim\b", c) or re.search(r"\b(?:no|missing) sim|sim card status", c):
            if re.search(r"\bpuk\b|locked_puk", c):
                put("sim", "locked_puk")
            elif re.search(r"\bpin\b|locked_pin|\blocked\b", c):
                put("sim", "locked_pin")
            elif re.search(r"\bno sim\b|sim (?:card )?(?:is |was )?(?:missing|not (?:detected|found|recognized|inserted)|absent)|missing|not detected|no sim card|\bsim\W+missing", c):
                put("sim", "missing")
            elif re.search(r"\b(?:active|working|fine|ok|okay|good|normal|detected|recognized)\b", c):
                put("sim", "active")
            if re.search(r"re-?seat\w* (?:successfully|ok|done)|successfully re-?seat|re-?seated successfully", c):
                put("sim_reseated", True)
        # speed test
        self._speed(c, put)
        # MMS
        self._mms(c, put, requested)
        # APN
        if re.search(r"\bapn\b|\bmmsc\b", c):
            if re.search(r"mmsc[^.]{0,25}(?:not set|empty|blank|missing|none|nothing)", c):
                put("apn_mmsc_set", False)
            elif re.search(r"mmsc[^.]{0,25}(?:set|http|https|www|url)", c):
                put("apn_mmsc_set", True)
            if re.search(r"apn(?: name)?[^.]{0,12}(?:not set|empty|blank|none)", c):
                put("apn_name_none", True)
            if re.search(r"will reset (?:at|on|after) (?:the )?(?:next )?reboot|reset (?:is )?pending|reset at reboot", c):
                put("apn_reset_pending", True)
        # app permissions
        self._perms(c, put)
        # payment
        self._payment(c, put)
        # wifi blocked by airplane
        if re.search(r"(?:cannot|can'?t|unable to) (?:change|turn|toggle)[^.]{0,20}wifi[^.]{0,25}airplane", c):
            put("wifi_blocked", True)
        # bare answers about the requested subject: "it's off"
        self._bare(c, put, requested)

    def _vpn(self, c: str, put) -> None:
        if not _RX["vpn"].search(c):
            return
        if re.search(r"no active vpn|no vpn connection to disconnect", c):
            put("vpn_action", "none_active")
            put("vpn", False)
        elif re.search(r"vpn (?:was |is )?disconnected(?: successfully)?|disconnected (?:from )?(?:the |my )?vpn|(?:vpn )?disconnect\w* successful", c):
            put("vpn_action", "disconnected")
            put("vpn", False)
        elif re.search(r"already connected", c):
            put("vpn_action", "already_connected")
            put("vpn", True)
        elif re.search(r"vpn connected successfully|connected (?:to )?(?:the |my )?vpn successfully", c):
            put("vpn_action", "connected")
            put("vpn", True)
        elif re.search(r"(?:vpn|it)[^.]{0,20}\b(?:on|enabled|turned on)\b[^.]{0,12}(?:but|not|isn'?t)[^.]{0,12}connected|turned on in settings", c):
            put("vpn_setting", True)
            put("vpn", False)
        elif re.search(r"\bno vpn\b|not (?:using|connected to|running) (?:a |any |the )?vpn|(?:don'?t|do not) (?:have|use) (?:a |any )?vpn|vpn (?:is |status is )?(?:off|disabled|turned off|not connected|disconnected)", c):
            put("vpn", False)
            put("vpn_setting", False)
        elif re.search(r"(?:don'?t|do not|can'?t|cannot|doesn'?t|not)\b[^.]{0,15}\b(?:see|seeing|have|notice|showing|shown|visible)\b[^.]{0,20}vpn", c):
            put("vpn", False)
        elif re.search(r"vpn (?:is )?(?:on|connected|active|running|enabled)|connected to (?:a |the |my )?vpn|(?:see|there'?s|there is|shows?)[^.]{0,20}vpn", c):
            put("vpn", True)
            put("vpn_setting", True)
        pm = re.search(r"\b(poor|fair|good|excellent|unknown)\b[^.]{0,25}(?:performance|server)|(?:performance|server)[^.]{0,25}\b(poor|fair|good|excellent|unknown)\b", c)
        if pm:
            put("vpn_perf", pm.group(1) or pm.group(2))

    def _speed(self, c: str, put) -> None:
        mb = re.search(r"(\d+(?:\.\d+)?)\s*(?:mbps|mb/s|megabits?)", c)
        spctx = bool(mb) or re.search(r"\bspeed\b|\btest\b|\bresult\b|came back", c)
        if mb:
            put("mbps", float(mb.group(1)))
            put("speed_ok", True)
        if re.search(r"no connection|speed test (?:failed|fails|won'?t run|did not run|didn'?t run)|test failed|(?:couldn'?t|can'?t|unable to|cannot) (?:run|complete|do) (?:the |a )?speed", c):
            put("speed_ok", False)
            put("speed_desc", "No Connection")
        elif spctx:
            d = re.search(r"\b(very poor|poor|fair|good|excellent)\b", c)
            if d and not (re.search(r"\b(?:signal|bars?)\b", c) and not mb and not re.search(r"speed|test", c)):
                put("speed_desc", " ".join(w.capitalize() for w in d.group(1).split()))
            elif mb:  # tau2's thresholds (user_tools._run_speed_test docstring): the description is a function of Mbps
                x = float(mb.group(1))
                put("speed_desc", "Very Poor" if x < 1 else "Poor" if x < 5 else "Fair" if x < 25 else "Good" if x < 100 else "Excellent")

    def _mms(self, c: str, put, requested: Sequence[DeviceAction]) -> None:
        obj = r"(?:mms|pictures?|photos?|images?|pics?|picture messages?|photo messages?|multimedia(?: messages?)?|attachments?)"
        if re.search(rf"\bbut\s+(?:not|can'?t|cannot|won'?t)\b[^.]{{0,25}}{obj}", c) or \
           re.search(rf"\b(?:can'?t|cannot|can not|unable to|not able to|couldn'?t|won'?t|still not|didn'?t|did not|doesn'?t|isn'?t|not)\b[^.]{{0,25}}\b(?:send|sending|sent|go through|goes through|went through|deliver\w*|work)\b[^.]{{0,25}}{obj}", c) or \
           re.search(rf"{obj}[^.]{{0,25}}\b(?:fail\w*|didn'?t (?:go|send)|won'?t (?:go|send)|not (?:sending|sent|going|working)|error)\b", c):
            put("mms", False)
        elif re.search(rf"\b(?:can|could|able to|was able to|managed to|successfully|did)\b[^.]{{0,12}}\b(?:send|sent|sending)\b[^.]{{0,25}}{obj}|"
                       rf"{obj}[^.]{{0,25}}\b(?:sent|went through|goes through|working|worked|success\w*)\b|\bmms (?:test )?(?:passed|ok)\b", c):
            put("mms", True)
        elif any(a.tool == "can_send_mms" for a in requested):
            if re.search(r"\b(?:yes|yeah|it worked|worked|works|success\w*|went through|sent)\b", c) and not re.search(r"\b(?:not|no|n't|fail\w*)\b", c):
                put("mms", True)
            elif re.search(r"\b(?:no|nope|didn'?t|did not|failed|fail|still not|not work\w*|doesn'?t work)\b", c):
                put("mms", False)

    def _perms(self, c: str, put) -> None:
        if not re.search(r"permission|\bgranted\b|\bapp\b|messaging", c):
            return
        if re.search(r"(?:app )?'?\w+'? (?:was )?not found|isn'?t installed|not installed", c):
            put("perm_found", False)
        listed = [p for p in _PERMS if re.search(rf"\b{p}\b", re.sub(r"\b(?:messaging|sms) app\b", " ", c))]
        if re.search(r"success\w*[^.]{0,20}(?:permission|granted)|permission[^.]{0,30}granted to|granted successfully", c):
            put("grant_ok", True)
        elif re.search(r"\berror\b[^.]{0,30}(?:permission|grant)|(?:couldn'?t|could not|can'?t|unable to|failed to)[^.]{0,15}grant", c):
            put("grant_ok", False)
        if re.search(r"no permissions?(?: granted)?|none granted|doesn'?t have any permissions|has no permissions|currently has no permissions", c):
            put("perm_found", True)
            for p in _PERMS:
                put(f"perm.{p}", False)
            put("perms_all", ())
            return
        if not listed:
            return
        negs = re.search(r"(?:doesn'?t|does not|don'?t|do not|isn'?t|not|missing|lacks?|without|no)\b[^.]{0,25}", c)
        if negs and re.search(r"(?:doesn'?t have|does not have|isn'?t granted|not granted|missing|lacks?|without|no|not been granted|denied)", c):
            seg = c[negs.start():]
            neg_perms = [p for p in _PERMS if re.search(rf"\b{p}\b", seg)]
            for p in neg_perms:
                put(f"perm.{p}", False)
            for p in listed:
                if p not in neg_perms:
                    put(f"perm.{p}", True)
            put("perm_found", True)
        elif re.search(r"permissions?|granted|has|have|access", c):
            put("perm_found", True)
            for p in listed:
                put(f"perm.{p}", True)
            if re.search(r"\bonly\b|\bjust\b|permission for:?|permissions? granted:?", c):
                for p in _PERMS:
                    if p not in listed:
                        put(f"perm.{p}", False)
                put("perms_all", tuple(sorted(listed)))

    def _payment(self, c: str, put) -> None:
        if re.search(r"no payment request|(?:don'?t|do not) have (?:a |any )?payment request|haven'?t (?:got|received) (?:a |any )?payment request", c):
            put("pay_pending", False)
            put("pay_paid", False) if re.search(r"do not have a payment request|do(?:n'?t| not) have a payment", c) else None
        elif re.search(r"payment request|bill", c) and re.search(r"\b(?:have|got|received|see|there'?s|there is|shows?|says)\b", c):
            put("pay_pending", True)
        am = re.search(r"\$\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:usd|dollars)", c)
        if am and re.search(r"payment|bill|request|due|usd|pay", c):
            put("pay_amount", float(am.group(1) or am.group(2)))
        bm = re.search(r"\bbill\s+(?:id\s+)?(?:is\s+)?#?([a-z]*\d{2,})\b", c)
        if bm:
            put("pay_bill", bm.group(1))
        if re.search(r"payment[^.]{0,25}(?:went through|was successful|has been made|was made|is done|made|processed|completed)|(?:i|just)\s+paid|paid (?:the|it|that)|payment successful", c):
            put("pay_paid", True)

    def _bare(self, c: str, put, requested: Sequence[DeviceAction]) -> None:
        """'it's off' / 'yes it's on' right after a single-subject request: bind the state to that subject."""
        m = re.search(r"\b(?:it|that)(?:'s| is| was now| is now)\s+(?:now\s+|currently\s+|still\s+)?(on|off|enabled|disabled)\b|^(on|off|enabled|disabled)\W*$", c)
        if not m or not requested:
            return
        subj = {a.tool for a in requested}
        key = {"toggle_airplane_mode": "airplane", "toggle_data": "data", "toggle_roaming": "roaming", "toggle_data_saver_mode": "data_saver",
               "toggle_wifi": "wifi_on", "toggle_wifi_calling": "wifi_calling", "check_wifi_calling_status": "wifi_calling",
               "check_data_restriction_status": "data_saver", "check_wifi_status": "wifi_on"}
        keys = {key[t] for t in subj if t in key}
        if len(keys) == 1:
            w = m.group(1) or m.group(2)
            put(next(iter(keys)), w in ("on", "enabled"))

    # ---- fact -> field mapping + closed-space filtering
    @staticmethod
    def _map(facts: dict[str, Any]) -> dict[str, Any]:
        f: dict[str, Any] = {}
        for k, v in facts.items():
            if k == "airplane":
                f.update({"status_bar.airplane_mode": v, "network.airplane_mode": v, "airplane_mode.now": v})
            elif k == "signal":
                f.update({"status_bar.signal": v, "network.signal": v})
            elif k == "signal_hidden":
                f.update({"status_bar.signal": "hidden", "status_bar.tech": "hidden", "status_bar.data": "hidden"})
            elif k == "conn":
                f["network.connection"] = v
            elif k == "tech":
                f.update({"status_bar.tech": v, "network.tech": v})
            elif k == "mode":
                f.update({"network_mode.preference": v, "network_mode.set": v})
            elif k == "mode_invalid":
                f["network_mode.invalid"] = v
            elif k == "data":
                f.update({"status_bar.data": "enabled" if v else "disabled", "network.data_enabled": v, "data.now": v})
            elif k == "data_saver":
                f.update({"status_bar.data_saver": v, "data_restriction.data_saver": v, "data_saver.now": v})
            elif k == "roaming":
                f.update({"network.roaming_enabled": v, "roaming.now": v})
            elif k == "wifi_on":
                f.update({"wifi.enabled": v, "network.wifi_enabled": v, "wifi.now": v})
            elif k == "wifi_conn":
                f.update({"wifi.connected": v, "network.wifi_connected": v, "status_bar.wifi": "connected" if v else "none"})
            elif k == "wifi_blocked":
                f["wifi.blocked_by_airplane"] = v
            elif k == "wifi_calling":
                f.update({"wifi_calling.enabled": v, "wifi_calling.now": v})
            elif k == "vpn":
                f.update({"status_bar.vpn": v, "vpn.connected": v})
            elif k == "vpn_setting":
                f["vpn.enabled_setting"] = v
            elif k == "vpn_perf":
                f["vpn.performance"] = v
            elif k == "vpn_action":
                f["vpn.action"] = v
            elif k == "sim":
                f.update({"sim.status": v, "network.sim_status": v})
            elif k == "sim_reseated":
                f["sim.reseated"] = v
            elif k == "mms":
                f["mms.can_send"] = v
            elif k == "mbps":
                f["speed_test.mbps"] = v
            elif k == "speed_ok":
                f["speed_test.ok"] = v
            elif k == "speed_desc":
                f["speed_test.desc"] = v
            elif k == "apn_mmsc_set":
                f["apn.mmsc_set"] = v
            elif k == "apn_name_none":
                f["apn.name"] = None
            elif k == "apn_reset_pending":
                f["apn.reset_pending"] = v
            elif k.startswith("perm."):
                p = k.split(".", 1)[1]
                f.update({f"app_permissions.perm.{p}": v, f"app_status.perm.{p}": v})
            elif k == "perms_all":
                f.update({"app_permissions.permissions": v, "app_status.permissions": v})
            elif k == "perm_found":
                f.update({"app_permissions.found": v, "app_status.found": v})
            elif k == "grant_ok":
                f["app_permission.grant_ok"] = v
            elif k == "pay_pending":
                f["payment_request.pending"] = v
                if v is False:
                    f["payment_request.amount"] = None
                    f["payment_request.bill_id"] = None
            elif k == "pay_amount":
                f.update({"payment_request.amount": v, "make_payment.amount": v})
            elif k == "pay_bill":
                f.update({"payment_request.bill_id": v, "make_payment.bill_id": v})
            elif k == "pay_paid":
                f["make_payment.paid"] = v
        return f

    @staticmethod
    def _in_space(v: Any, allowed: Sequence[Any]) -> Any:
        """The allowed spelling of `v` (type-strict for bool, case-insensitive for str, tolerant for floats) or _MISS."""
        for a in allowed:
            if isinstance(v, bool) or isinstance(a, bool):
                if isinstance(v, bool) and isinstance(a, bool) and v == a:
                    return a
            elif v is None or a is None:
                if v is None and a is None:
                    return a
            elif isinstance(v, (int, float)) and isinstance(a, (int, float)):
                if math.isclose(float(v), float(a), abs_tol=0.005):
                    return a
            elif isinstance(v, str) and isinstance(a, str):
                if v.casefold() == a.casefold():
                    return a
            elif v == a:
                return a
        return _MISS

    def _literals(self, text: str, label_space: Mapping[str, Sequence[Any]], fields: dict, out_conf: list) -> None:
        """Free-form string fields (ssid, protocol, APN name, ...): emitted only when an allowed value appears verbatim."""
        ctx = {"apn.name": r"\bapn\b", "apn.mmsc_url": r"\bmmsc\b|\burl\b", "vpn.protocol": r"\bvpn\b|protocol", "vpn.server_address": r"\bvpn\b|server",
               "status_bar.wifi_ssid": r"wifi|network|connected", "network.wifi_ssid": r"wifi|network|connected",
               "wifi.ssid": r"wifi|network|connected", "app_status.app": r"\bapp\b|status", "app_permissions.app": r"\bapp\b|permission"}
        for fld, cx in ctx.items():
            allowed = label_space.get(fld)
            if not allowed or fld in fields:
                continue
            hits = []
            for sent in (text[a:b] for a, b in _sentence_spans(text)):
                if not re.search(cx, sent):
                    continue
                for a in allowed:
                    if isinstance(a, str) and len(a) >= 3 and re.search(rf"(?<![\w]){re.escape(a.casefold())}(?![\w])", sent) and a not in hits:
                        hits.append(a)
            if len(hits) == 1:
                fields[fld] = hits[0]
            elif len(hits) > 1:
                out_conf.append((fld, tuple(hits)))

    def parse(self, user_text: str, requested: tuple, label_space: Mapping[str, Sequence[Any]]) -> ParsedObservation:
        text = _norm(user_text)
        facts = self._facts(text, requested)
        conflicts: list = []
        out_space: list = []
        fields: dict[str, Any] = {}
        resolved: dict[str, Any] = {}
        for k, vals in facts.items():
            distinct = list(dict.fromkeys(vals))
            if len(distinct) > 1:
                for fld in self._map({k: distinct[0]}):
                    if fld in label_space:
                        conflicts.append((fld, tuple(distinct)))
                continue
            resolved[k] = distinct[0]
        if resolved.get("wifi_on") is False and "wifi_conn" not in resolved and "wifi_conn" not in facts:
            resolved["wifi_conn"] = False
        if resolved.get("airplane") is True and any(k in resolved for k in ()):
            pass
        for fld, v in self._map(resolved).items():
            if fld not in label_space:
                continue
            ok = self._in_space(v, label_space[fld])
            if ok is _MISS:
                out_space.append((fld, v))
            else:
                fields[fld] = ok
        self._literals(text, label_space, fields, conflicts)
        conflict_fields = {c[0] for c in conflicts}
        fields = {k: v for k, v in fields.items() if k not in conflict_fields}
        return ParsedObservation(actions=tuple(requested), source="user_report", fields=_frozen_items(fields),
                                 mentioned=frozenset(fields), raw=user_text, conflicts=tuple(sorted(conflicts, key=lambda x: x[0])),
                                 out_of_space=tuple(sorted(out_space, key=lambda x: x[0])))


_MISS = object()


# =========================================================================== LLM-backed stubs + factory
_LLM_MSG = ("The LLM-backed parser is not implemented yet (planned for after the rule parsers are measured on the "
            "logged train split, W4). Set config `parser: rule` or use telecom_boed.belief.parsers.build_parsers('rule').")


class LLMActionParser:
    def parse(self, agent_text: str) -> tuple[DeviceAction, ...]:
        raise NotImplementedError(_LLM_MSG)


class LLMObservationParser:
    def parse(self, user_text: str, requested: tuple, label_space: Mapping[str, Sequence[Any]]) -> Observation:
        raise NotImplementedError(_LLM_MSG)


class LLMClaimClassifier:
    def prob_claim(self, agent_text: str) -> float:
        raise NotImplementedError(_LLM_MSG)


class LLMIntentParser:
    def parse(self, first_user_text: str) -> dict[str, float]:
        raise NotImplementedError(_LLM_MSG)


@dataclass(frozen=True)
class ParserBundle:
    action: Any
    observation: Any
    claim: Any
    intent: Any


def build_parsers(kind: str = "rule") -> ParserBundle:
    """Config switch `parser: rule|llm`. 'llm' returns stubs that raise NotImplementedError when called."""
    if kind == "rule":
        return ParserBundle(RuleActionParser(), RuleObservationParser(), RuleClaimClassifier(), RuleIntentParser())
    if kind == "llm":
        return ParserBundle(LLMActionParser(), LLMObservationParser(), LLMClaimClassifier(), LLMIntentParser())
    raise ValueError(f"unknown parser kind {kind!r} (expected 'rule' or 'llm')")
