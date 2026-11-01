"""[telecom_boed] Field-level featurization of tau2 telecom tool outputs (owner: W1-S2). Pure text/JSON; no tau2 import.

featurize(tool, content) -> {"<group>.<field>": value}. Values are bool / int / float / str / None / tuple (hashable).
Every format is parsed from its source line (tau2 `src/tau2/domains/telecom/`, `uf` = user_tools.py, `tf` = tools.py).
- Errors (content starts "Error: ", environment.py get_response) -> {"error": True, "error.msg": ...}.
- Unknown tool or an unparseable content -> {"<tool>.raw": content}  (the test asserts no known tool ever lands here).
- Write tools return "<status line(s)>\nStatus Bar: <bar>": the bar is parsed into status_bar.* and the status line(s)
  are kept verbatim as "<tool>.msg" next to the tool's structured fields.
- status_bar.* is what the phone DISPLAYS: fields the bar hides (airplane mode hides signal / tech / data icon) are the
  string "hidden"; status_bar.data_saver is only shown with an enabled data icon, so False there means "not shown".
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Iterable, Optional

Feats = dict[str, Any]
_ERR = "Error: "


def _b(onoff: str) -> bool:
    return onoff.strip().upper() in ("ON", "YES")


# --------------------------------------------------------------------------- status bar (uf:76-139)
_SIGNAL = {"No Signal": "none", "Poor": "poor", "Fair": "fair", "Good": "good", "Excellent": "excellent"}


def _status_bar(content: str) -> Feats:
    """`Status Bar: ✈️ Airplane Mode | 🔋 80%` or `Status Bar: 📶³ Good | 5G | 📱 Data Enabled | 🔽 Data Saver |
    📡 Connected to <ssid> | 🔒 VPN Connected | 🔋 80%` (uf:81-139; `|`-joined, order fixed)."""
    body = content.split("Status Bar:", 1)[1].strip()
    items = [i.strip() for i in body.split(" | ")]
    f: Feats = {"status_bar.airplane_mode": False, "status_bar.signal": "hidden", "status_bar.tech": "hidden",
                "status_bar.data": "hidden", "status_bar.data_saver": False, "status_bar.wifi": "none",
                "status_bar.wifi_ssid": None, "status_bar.vpn": False, "status_bar.battery": None}
    seen_signal = False
    for it in items:
        if "Airplane Mode" in it:
            f["status_bar.airplane_mode"] = True
        elif "No Signal" in it or it.startswith("📶"):
            word = "No Signal" if "No Signal" in it else it.split(" ", 1)[1].strip()
            f["status_bar.signal"] = _SIGNAL[word]
            seen_signal = True
        elif re.fullmatch(r"[2-5]G", it):
            f["status_bar.tech"] = it
        elif "Data Enabled" in it:
            f["status_bar.data"] = "enabled"
        elif "Data Disabled" in it:
            f["status_bar.data"] = "disabled"
        elif "Data Saver" in it:
            f["status_bar.data_saver"] = True
        elif it.startswith("📡"):
            m = re.fullmatch(r"📡 Connected to (.*)", it)
            f["status_bar.wifi"], f["status_bar.wifi_ssid"] = ("connected", m.group(1)) if m else ("enabled", None)
        elif "VPN Connected" in it:
            f["status_bar.vpn"] = True
        elif "🔋" in it:
            f["status_bar.battery"] = int(re.search(r"(\d+)%", it).group(1))
    if seen_signal and f["status_bar.tech"] == "hidden":
        f["status_bar.tech"] = "none"  # signal icon shown but no technology item
    return f


def _split_bar(content: str) -> tuple[str, Feats]:
    """(status text before the bar, status_bar.* features) for write tools; no bar -> ({content}, {})."""
    if "\nStatus Bar:" in content:
        msg, bar = content.split("\nStatus Bar:", 1)
        return msg, _status_bar("Status Bar:" + bar)
    return content, {}


def _write(group: str = "", pat: Optional[str] = None, fields: Optional[Callable[[re.Match], Feats]] = None) -> Callable[[str, str], Feats]:
    """Write-tool parser: status text -> <tool>.msg (+ regex-derived fields), trailing Status Bar -> status_bar.*."""
    def parse(tool: str, content: str) -> Feats:
        msg, bar = _split_bar(content)
        f: Feats = {f"{tool}.msg": msg}
        if pat is not None:
            m = re.search(pat, msg, re.S)
            if not m:
                raise ValueError(f"{tool}: unexpected format")
            f.update(fields(m) if fields else {})
        f.update(bar)
        return f
    return parse


def _now(group: str) -> Callable[[str, str], Feats]:
    """`<X> is now ON|OFF.` toggles (uf:372-373 airplane, 474-478 data, 505-506 roaming, 549-553 saver, 656 wifi,
    693-697 wifi calling)."""
    return _write(group, r"is now (ON|OFF)", lambda m: {f"{group}.now": m.group(1) == "ON"})


# --------------------------------------------------------------------------- read tools
def _network_status(tool: str, c: str) -> Feats:
    """9-10 `Key: Value` lines (uf:146-159): Airplane Mode ON|OFF; SIM Card Status <enum>; Cellular Connection <enum>;
    Cellular Signal <enum>; Cellular Network Type <enum>; Mobile Data Enabled Yes|No; Data Roaming Enabled Yes|No;
    Wi-Fi Radio ON|OFF; Wi-Fi Connected Yes|No; [Connected Wi-Fi Network <ssid>]."""
    kv = dict(line.split(": ", 1) for line in c.splitlines())
    return {"network.airplane_mode": _b(kv["Airplane Mode"]), "network.sim_status": kv["SIM Card Status"],
            "network.connection": kv["Cellular Connection"], "network.signal": kv["Cellular Signal"],
            "network.tech": kv["Cellular Network Type"], "network.data_enabled": _b(kv["Mobile Data Enabled"]),
            "network.roaming_enabled": _b(kv["Data Roaming Enabled"]), "network.wifi_enabled": _b(kv["Wi-Fi Radio"]),
            "network.wifi_connected": _b(kv["Wi-Fi Connected"]), "network.wifi_ssid": kv.get("Connected Wi-Fi Network")}


def _speed(tool: str, c: str) -> Feats:
    """`Speed Test Result: 27.50 Mbps (Good). <advice>` or `Speed test failed: <reason>.` (uf:250-270)."""
    m = re.match(r"Speed Test Result: ([\d.]+) Mbps \(([^)]*)\)\.", c)
    if m:
        return {"speed_test.ok": True, "speed_test.mbps": float(m.group(1)), "speed_test.desc": m.group(2)}
    m = re.match(r"Speed test failed: (.*)\.$", c)
    if m:
        return {"speed_test.ok": False, "speed_test.mbps": None, "speed_test.desc": m.group(1)}
    raise ValueError("speed")


_SIM = {"Your SIM card is active and working.": "active", "No SIM card detected in the phone.": "missing",
        "The SIM card is locked with a PIN code.": "locked_pin", "The SIM card is locked with a PUK code.": "locked_puk"}


def _sim(tool: str, c: str) -> Feats:
    """One sentence per state (uf:420-427)."""
    return {"sim.status": _SIM[c]}


def _apn(tool: str, c: str) -> Feats:
    """`Current APN Name: <name|Not Set>\\nMMSC URL (for picture messages): <url|Not Set>\\n(...)` (uf:580-583)."""
    name = re.search(r"Current APN Name: (.*)", c).group(1)
    url = re.search(r"MMSC URL \(for picture messages\): (.*)", c).group(1)
    return {"apn.name": None if name == "Not Set" else name, "apn.mmsc_set": url != "Not Set",
            "apn.mmsc_url": None if url == "Not Set" else url}


def _wifi(tool: str, c: str) -> Feats:
    """`Wi-Fi is turned OFF.` | `Wi-Fi is ON but not connected to any network.` |
    `Wi-Fi is ON and connected to '<ssid>'. Signal strength: <enum>.` (uf:633-639)."""
    if c.startswith("Wi-Fi is turned OFF"):
        return {"wifi.enabled": False, "wifi.connected": False, "wifi.ssid": None, "wifi.signal": None}
    if "not connected" in c:
        return {"wifi.enabled": True, "wifi.connected": False, "wifi.ssid": None, "wifi.signal": None}
    m = re.match(r"Wi-Fi is ON and connected to '(.*)'\. Signal strength: (\w+)\.", c)
    return {"wifi.enabled": True, "wifi.connected": True, "wifi.ssid": m.group(1), "wifi.signal": m.group(2)}


def _vpn(tool: str, c: str) -> Feats:
    """`VPN is turned OFF.` | `VPN is turned ON in settings, but currently not connected.` |
    `VPN is ON and connected. Details: {'server_address': .., 'protocol': .., 'server_performance':
    <PerformanceLevel.POOR: 'poor'>}` | `VPN is ON and connected (no specific details available).` (uf:721-731)."""
    f: Feats = {"vpn.connected": False, "vpn.enabled_setting": False, "vpn.server_address": None,
                "vpn.protocol": None, "vpn.performance": None}
    if c.startswith("VPN is ON and connected"):
        f["vpn.connected"] = f["vpn.enabled_setting"] = True
        for k, pat in (("server_address", r"'server_address': '([^']*)'"), ("protocol", r"'protocol': '([^']*)'"),
                       ("performance", r"'server_performance': <PerformanceLevel\.\w+: '(\w+)'>")):
            m = re.search(pat, c)
            f[f"vpn.{k}"] = m.group(1) if m else None
    elif c.startswith("VPN is turned ON"):
        f["vpn.enabled_setting"] = True
    elif not c.startswith("VPN is turned OFF"):
        raise ValueError("vpn")
    return f


_PERMS = ("sms", "storage", "phone", "network")  # AppPermissions fields (user_data_model.py:115-123)


def _perm_feats(group: str, perms: tuple[str, ...], found: bool, app: Optional[str]) -> Feats:
    f: Feats = {f"{group}.found": found, f"{group}.app": app, f"{group}.permissions": perms}
    for p in _PERMS:
        f[f"{group}.perm.{p}"] = (p in perms) if found else None
    return f


def _app_status(tool: str, c: str) -> Feats:
    """`Status for App: <app>\\n - Permissions Granted:\\n   - sms\\n ...` | ` - Permissions: None granted.` |
    `App '<app>' not found on this phone.` (uf:808-826)."""
    m = re.match(r"App '(.*)' not found", c)
    if m:
        return _perm_feats("app_status", (), False, m.group(1))
    app = re.match(r"Status for App: (.*)", c).group(1)
    perms = tuple(sorted(re.findall(r"^\s+- (\w+)$", c, re.M)))
    return _perm_feats("app_status", perms, True, app)


def _app_perms(tool: str, c: str) -> Feats:
    """`App '<app>' has permission for: sms, storage.` | `App '<app>' currently has no permissions granted.` |
    `App '<app>' not found on this phone.` (uf:839-851)."""
    m = re.match(r"App '(.*)' not found", c)
    if m:
        return _perm_feats("app_permissions", (), False, m.group(1))
    m = re.match(r"App '(.*)' has permission for: (.*)\.$", c)
    if m:
        return _perm_feats("app_permissions", tuple(sorted(x.strip() for x in m.group(2).split(","))), True, m.group(1))
    m = re.match(r"App '(.*)' currently has no permissions granted", c)
    return _perm_feats("app_permissions", (), True, m.group(1))


def _payment_request(tool: str, c: str) -> Feats:
    """`No payment request has been made.` | `You have a payment request for bill <id> of <amount> USD.` (uf:1056-1062)."""
    m = re.match(r"You have a payment request for bill (\S+) of ([\d.]+) USD", c)
    if m:
        return {"payment_request.pending": True, "payment_request.bill_id": m.group(1),
                "payment_request.amount": float(m.group(2))}
    if c.startswith("No payment request"):
        return {"payment_request.pending": False, "payment_request.bill_id": None, "payment_request.amount": None}
    raise ValueError("payment_request")


def _make_payment(tool: str, c: str) -> Feats:
    """`Payment of <amount> USD has been made for bill <id>.` | `You do not have a payment request.` (uf:1074-1090)."""
    m = re.match(r"Payment of ([\d.]+) USD has been made for bill (\S+?)\.$", c)
    if m:
        return {"make_payment.paid": True, "make_payment.amount": float(m.group(1)), "make_payment.bill_id": m.group(2)}
    if c.startswith("You do not have a payment request"):
        return {"make_payment.paid": False, "make_payment.amount": None, "make_payment.bill_id": None}
    raise ValueError("make_payment")


def _vpn_action(tool: str, c: str) -> Feats:
    """connect (uf:746-756): `VPN connected successfully.` | `VPN already connected.` (no bar) | `No VPN connection to
    connect.`; disconnect (uf:768-777): `VPN disconnected successfully.` | `No active VPN connection to disconnect.`"""
    msg, bar = _split_bar(c)
    res = {"VPN connected successfully.": "connected", "VPN already connected.": "already_connected",
           "No VPN connection to connect.": "unavailable", "VPN disconnected successfully.": "disconnected",
           "No active VPN connection to disconnect.": "none_active"}[msg]
    return {f"{tool}.msg": msg, "vpn.action": res, **bar}


def _grant(tool: str, c: str) -> Feats:
    """`Success. Permission '<p>' granted to app '<a>'.` | `Error. <reason>` + Status Bar (uf:860-870)."""
    msg, bar = _split_bar(c)
    return {"grant_app_permission.msg": msg, "app_permission.grant_ok": msg.startswith("Success."), **bar}


def _reboot(tool: str, c: str) -> Feats:
    """`[Resetting APN settings...\\n]Restarting network services...` + Status Bar (uf:947-964)."""
    msg, bar = _split_bar(c)
    return {"reboot_device.msg": msg, "reboot.apn_reset": "Resetting APN settings" in msg, **bar}


def _wifi_toggle(tool: str, c: str) -> Feats:
    """`Wi-Fi is now ON|OFF.` | `Cannot change Wi-Fi settings while Airplane Mode is ON.` + Status Bar (uf:652-657)."""
    msg, bar = _split_bar(c)
    blocked = msg.startswith("Cannot change")
    return {"toggle_wifi.msg": msg, "wifi.now": None if blocked else msg.startswith("Wi-Fi is now ON"),
            "wifi.blocked_by_airplane": blocked, **bar}


# --------------------------------------------------------------------------- agent tools (JSON, tf = tools.py)
def _customer(d: dict) -> Feats:
    """Customer dump (data_model.py:176-209)."""
    return {"customer.id": d["customer_id"], "customer.account_status": d["account_status"],
            "customer.line_ids": tuple(d["line_ids"]), "customer.bill_ids": tuple(d["bill_ids"]),
            "customer.name": d["full_name"], "customer.phone_number": d["phone_number"]}


def _line(d: dict) -> Feats:
    """Line dump (data_model.py:69-104)."""
    return {"line.id": d["line_id"], "line.status": d["status"], "line.plan_id": d["plan_id"],
            "line.device_id": d["device_id"], "line.data_used_gb": float(d["data_used_gb"]),
            "line.data_refueling_gb": float(d["data_refueling_gb"]), "line.roaming_enabled": d["roaming_enabled"],
            "line.contract_end_date": d["contract_end_date"], "line.suspension_start_date": d["suspension_start_date"]}


def _bill(d: dict) -> Feats:
    """Bill dump (data_model.py:128-150)."""
    return {"bill.id": d["bill_id"], "bill.status": d["status"], "bill.total_due": float(d["total_due"]),
            "bill.due_date": d["due_date"], "bill.n_items": len(d["line_items"])}


def _plan(d: dict) -> Feats:
    """Plan dump (data_model.py:21-30)."""
    return {"plan.id": d["plan_id"], "plan.name": d["name"], "plan.data_limit_gb": float(d["data_limit_gb"]),
            "plan.price_per_month": float(d["price_per_month"]),
            "plan.refuel_price_per_gb": float(d["data_refueling_price_per_gb"])}


def _device(d: dict) -> Feats:
    """Device dump (data_model.py:39-60)."""
    return {"device.id": d["device_id"], "device.type": d["device_type"], "device.activated": d["activated"],
            "device.is_esim_capable": d["is_esim_capable"]}


def _by_keys(d: dict) -> Feats:
    """get_details_by_id (tf:234-259) returns a Customer | Line | Bill | Plan | Device dump; dispatch on distinguishing keys."""
    if "account_status" in d:
        return _customer(d)
    if "bill_id" in d:
        return _bill(d)
    if "line_id" in d:
        return _line(d)
    if "device_type" in d:
        return _device(d)
    if "data_limit_gb" in d:
        return _plan(d)
    raise ValueError("details")


def _json(fn: Callable[[Any], Feats]) -> Callable[[str, str], Feats]:
    return lambda tool, c: fn(json.loads(c))


def _customers(d: list) -> Feats:
    """get_customer_by_name (tf:90-111): list of Customer dumps; first one featurized, count in customers.n."""
    f: Feats = {"customers.n": len(d)}
    if d:
        f.update(_customer(d[0]))
    return f


def _bills(d: list) -> Feats:
    """get_bills_for_customer (tf:335-358): newest-first list of Bill dumps."""
    st = tuple(b["status"] for b in d)
    return {"bills.n": len(d), "bills.ids": tuple(b["bill_id"] for b in d), "bills.statuses": st,
            "bills.total_due": tuple(float(b["total_due"]) for b in d),
            "bills.n_overdue": st.count("Overdue"), "bills.has_awaiting_payment": "Awaiting Payment" in st}


def _usage(d: dict) -> Feats:
    """get_data_usage (tf:485-515): values arrive as strings. exceeded mirrors environment.py:70
    (used >= limit + refueling)."""
    used, lim, ref = float(d["data_used_gb"]), float(d["data_limit_gb"]), float(d["data_refueling_gb"])
    return {"data_usage.line_id": d["line_id"], "data_usage.used_gb": used, "data_usage.limit_gb": lim,
            "data_usage.refueling_gb": ref, "data_usage.cycle_end_date": d["cycle_end_date"],
            "data_usage.exceeded": used >= lim + ref}


def _line_action(d: dict) -> Feats:
    """suspend_line / resume_line (tf:262-333): {"message": .., "line": Line dump}."""
    return {"line_action.message": d["message"], **_line(d["line"])}


def _refuel(d: dict) -> Feats:
    """refuel_data (tf:608-657): {"message": .., "new_data_refueling_gb": "2.0", "charge": "4.0"}."""
    return {"refuel.message": d["message"], "refuel.new_refueling_gb": float(d["new_data_refueling_gb"]),
            "refuel.charge": float(d["charge"])}


_ROAM = {"Roaming enabled successfully": "enabled", "Roaming was already enabled": "already_enabled",
         "Roaming disabled successfully": "disabled", "Roaming was already disabled": "already_disabled"}

_PARSERS: dict[str, Callable[[str, str], Feats]] = {
    # user tools (30)
    "check_status_bar": lambda t, c: _status_bar(c),
    "check_network_status": _network_status,
    "check_network_mode_preference": lambda t, c: {"network_mode.preference": c.split("Preference: ", 1)[1]},  # uf:181
    "set_network_mode_preference": _write("network_mode", r"(?:Preferred Network Mode set to: (\S+)|Failed to set network mode)",
                                          lambda m: {"network_mode.set": m.group(1), "network_mode.invalid": m.group(1) is None}),
    "run_speed_test": _speed,
    "toggle_airplane_mode": _now("airplane_mode"),
    "check_sim_status": _sim,
    "reseat_sim_card": _write("sim", r"re-seated successfully", lambda m: {"sim.reseated": True}),  # uf:437-439
    "toggle_data": _now("data"),
    "toggle_roaming": _now("roaming"),
    "check_data_restriction_status": lambda t, c: {"data_restriction.data_saver": "mode is ON" in c},  # uf:532-537
    "toggle_data_saver_mode": _now("data_saver"),
    "check_apn_settings": _apn,
    "set_apn_settings": _write("apn", r"APN settings set to: (\S+)", lambda m: {"apn.set": m.group(1)}),  # uf:593-597
    "reset_apn_settings": _write("apn", r"will reset at reboot", lambda m: {"apn.reset_pending": True}),  # uf:606-609
    "check_wifi_status": _wifi,
    "toggle_wifi": _wifi_toggle,
    "check_wifi_calling_status": lambda t, c: {"wifi_calling.enabled": "turned ON" in c},  # uf:677-681
    "toggle_wifi_calling": _now("wifi_calling"),
    "check_vpn_status": _vpn,
    "connect_vpn": _vpn_action,
    "disconnect_vpn": _vpn_action,
    "check_installed_apps": lambda t, c: {"apps.installed": tuple(a.strip() for a in c.split("phone: ", 1)[1].split(",") if a.strip())},  # uf:795-798
    "check_app_status": _app_status,
    "check_app_permissions": _app_perms,
    "grant_app_permission": _grant,
    "can_send_mms": lambda t, c: {"mms.can_send": "can send MMS" in c},  # uf:906-912
    "reboot_device": _reboot,
    "check_payment_request": _payment_request,
    "make_payment": _make_payment,
    # agent tools (13)
    "get_customer_by_phone": _json(_customer),
    "get_customer_by_id": _json(_customer),
    "get_customer_by_name": _json(_customers),
    "get_details_by_id": _json(_by_keys),
    "suspend_line": _json(_line_action),
    "resume_line": _json(_line_action),
    "get_bills_for_customer": _json(_bills),
    "send_payment_request": lambda t, c: {"payment_request_sent.bill_id": re.search(r"bill (\S+)$", c).group(1)},  # tf:393
    "get_data_usage": _json(_usage),
    "enable_roaming": lambda t, c: {"roaming_action.result": _ROAM[c]},  # tf:557-563
    "disable_roaming": lambda t, c: {"roaming_action.result": _ROAM[c]},  # tf:583-589
    "transfer_to_human_agents": lambda t, c: {"transfer.ok": c.startswith("Transfer successful")},  # tf:605
    "refuel_data": _json(_refuel),
}
USER_TOOLS = tuple(list(_PARSERS)[:30])
AGENT_TOOLS = tuple(list(_PARSERS)[30:])


def featurize(tool: str, content: str) -> Feats:
    """Field-level features of one tool output (see module docstring)."""
    if content.startswith(_ERR):
        return {"error": True, "error.msg": content[len(_ERR):]}
    fn = _PARSERS.get(tool)
    if fn is None:
        return {f"{tool}.raw": content}
    try:
        return fn(tool, content)
    except (ValueError, KeyError, AttributeError, IndexError, TypeError, json.JSONDecodeError):
        return {f"{tool}.raw": content}


def _sortkey(v: Any) -> tuple:
    return (type(v).__name__, str(v)) if not isinstance(v, (int, float)) or isinstance(v, bool) else ("num", v)


def label_space(contents_by_particle: Iterable[str], tool: str) -> dict[str, list]:
    """Closed label space of `tool`'s parser over the possible outputs: {field: sorted distinct values}.
    Union over the supplied contents (one per particle / possible world); None and mixed types sort deterministically.
    A field missing from some outputs simply contributes nothing for those."""
    out: dict[str, set] = {}
    for c in contents_by_particle:
        for k, v in featurize(tool, c).items():
            out.setdefault(k, set()).add(v)
    return {k: sorted(vs, key=_sortkey) for k, vs in sorted(out.items())}
