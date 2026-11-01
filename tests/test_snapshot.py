"""[telecom_boed] W1-S1 tests: fast_env equivalence, exact snapshot/restore/branch, replay equality, VPN isolation,
deterministic bill ids, timings. Only TRAIN-split tasks are read (split_tasks.json key "train")."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.telecom.environment import get_environment
from tau2.environment.toolkit import get_tool_signatures

from telecom_boed import paths
from telecom_boed.env import snapshot as S
from telecom_boed.env import tau2_patches

FIXTURES = Path(__file__).parent / "fixtures" / "W1-S1"
TIMINGS = paths.PACKAGE_ROOT / "results" / "timings" / "W1-S1.json"
CUSTOMERS = ["C1001", "C1002", "C1003", "C1004", "C9999"]
LINES = [f"L100{i}" for i in range(1, 10)] + ["L9999"]
BILLS = [f"B100{i}" for i in range(1, 7)] + ["B9999"]
PHONES = ["555-123-2002", "555-123-2001", "555-000-0000"]


# --------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def train_tasks(catalog, train_ids) -> dict[str, dict]:
    by_id = {t["id"]: t for t in catalog}
    return {i: by_id[i] for i in train_ids}  # never touches the test split


def intent_of(task_id: str) -> str:
    return task_id[1:].split("]")[0]


def init_calls(task: dict) -> list[EnvFunctionCall]:
    return [EnvFunctionCall(**a) for a in task["initial_state"]["initialization_actions"]]


def init_env(task: dict):
    env = S.fast_env()
    for a in init_calls(task):
        env.run_env_function_call(a)
    return env


def full_key(env) -> tuple:
    return S.state_key(env)


def _call_specs():
    """tool name -> requestor, from the real tool signatures."""
    env = S.fast_env()
    return [(n, "assistant") for n in env.tools.tools] + [(n, "user") for n in env.user_tools.tools]


def _rand_args(rng: random.Random, name: str, requestor: str, env) -> dict:
    kit = env.tools if requestor == "assistant" else env.user_tools
    args: dict[str, Any] = {}
    for p in inspect.signature(kit.tools[name]).parameters.values():
        n = p.name
        if n == "customer_id":
            args[n] = rng.choice(CUSTOMERS[:3] * 3 + CUSTOMERS[3:])
        elif n == "line_id":
            args[n] = rng.choice(LINES[:3] * 3 + LINES[3:])
        elif n == "bill_id":
            args[n] = rng.choice(BILLS)
        elif n == "phone_number":
            args[n] = rng.choice(PHONES)
        elif n == "id":
            args[n] = rng.choice(CUSTOMERS + LINES + BILLS)
        elif n == "full_name":
            args[n] = "John Smith"
        elif n == "dob":
            args[n] = "1990-01-01"
        elif n == "reason":
            args[n] = "Contract ended"
        elif n == "limit":
            args[n] = rng.choice([1, 12])
        elif n == "gb_amount":
            args[n] = rng.choice([2.0, 0.5, -1.0])
        elif n == "summary":
            args[n] = "escalate"
        elif n == "mode":
            args[n] = rng.choice(["4g_5g_preferred", "4g_only", "3g_only", "2g_only", "bogus"])
        elif n == "apn_settings":
            args[n] = {"apn_name": rng.choice(["internet", "broken"]), "reset_at_reboot": rng.random() < 0.3,
                       "mms_apn": rng.choice(["mms", None])}
        elif n == "app_name":
            args[n] = rng.choice(["messaging", "browser", "nope"])
        elif n == "permission":
            args[n] = rng.choice(["sms", "storage", "phone", "network", "bogus"])
        else:
            raise AssertionError(f"no generator for parameter {name}.{n}")
    return args


def gen_messages(rng: random.Random, env, length: int) -> list[list[tuple[str, str, dict]]]:
    """Messages = lists of (requestor, tool, args). Total call count == length. Mixes reads, writes, errors,
    hallucinated names, multi-call messages and the payment flow."""
    specs = _call_specs()
    flows = [
        [("assistant", "send_payment_request", {"customer_id": "C1001", "bill_id": "B1002"}),
         ("user", "check_payment_request", {}), ("user", "make_payment", {})],
        [("assistant", "refuel_data", {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2.0})],
        [("assistant", "refuel_data", {"customer_id": "C1002", "line_id": "L1004", "gb_amount": 2.0})],
        [("user", "connect_vpn", {}), ("user", "run_speed_test", {})],
        [("user", "toggle_airplane_mode", {}), ("user", "toggle_data", {}), ("user", "toggle_wifi", {})],
        [("user", "toggle_roaming", {}), ("assistant", "enable_roaming", {"customer_id": "C1001", "line_id": "L1002"})],
        [("user", "disconnect_vpn", {}), ("user", "reboot_device", {})],
    ]
    calls: list[tuple[str, str, dict]] = []
    while len(calls) < length:
        r = rng.random()
        if r < 0.30:
            calls.extend(rng.choice(flows))
        elif r < 0.38:
            calls.append((rng.choice(["user", "assistant"]), rng.choice(["fix_everything", "toggle_bluetooth"]), {}))
        elif r < 0.46:
            name, req = rng.choice(specs)
            calls.append((req, name, {"bogus_arg": 1}))  # bad args
        else:
            name, req = rng.choice(specs)
            calls.append((req, name, _rand_args(rng, name, req, env)))
    calls = calls[:length]
    msgs: list[list[tuple[str, str, dict]]] = []
    i = 0
    while i < len(calls):
        n = 1
        if rng.random() < 0.25:  # multi-call message: consecutive calls with the same requestor
            while i + n < len(calls) and n < 3 and calls[i + n][0] == calls[i][0]:
                n += 1
        msgs.append(calls[i:i + n])
        i += n
    return msgs


class Recorder:
    """Runs calls on an env through get_response and records tau2 messages."""

    def __init__(self, env):
        self.env = env
        self.messages: list = []
        self.tool_msgs: list[ToolMessage] = []
        self.n_calls = 0

    def run_message(self, msg: list[tuple[str, str, dict]], on_call=None) -> None:
        tcs = [ToolCall(id=f"c{self.n_calls + j}", name=n, arguments=copy.deepcopy(a), requestor=req)
               for j, (req, n, a) in enumerate(msg)]
        cls = AssistantMessage if msg[0][0] == "assistant" else UserMessage
        self.messages.append(cls(role=msg[0][0], tool_calls=tcs))
        for tc in tcs:
            tm = self.env.get_response(tc)
            self.messages.append(tm)
            self.tool_msgs.append(tm)
            self.n_calls += 1
            if on_call:
                on_call(self.n_calls)


# --------------------------------------------------------------------------- tests
def test_fast_env_matches_get_environment():
    ref = get_environment()
    fast = S.fast_env()
    assert fast.policy == ref.policy
    assert fast.domain_name == ref.domain_name and fast.solo_mode == ref.solo_mode
    assert sorted(fast.tools.tools) == sorted(ref.tools.tools) and len(fast.tools.tools) == 13
    assert sorted(fast.user_tools.tools) == sorted(ref.user_tools.tools) and len(fast.user_tools.tools) == 30
    sig = lambda e: {k: v.model_dump() for k, v in get_tool_signatures(e.tools).items()}  # noqa: E731
    usig = lambda e: {k: v.model_dump() for k, v in get_tool_signatures(e.user_tools).items()}  # noqa: E731
    assert sig(fast) == sig(ref) and usig(fast) == usig(ref)
    assert fast.get_db_hash() == ref.get_db_hash()
    assert fast.get_user_db_hash() == ref.get_user_db_hash()
    assert S.state_key(fast) == S.state_key(ref)
    assert fast.tools.db is not fast.user_tools.db
    # two fast envs never share mutable state
    other = S.fast_env()
    other.user_tools.db.device.airplane_mode = not other.user_tools.db.device.airplane_mode
    assert S.state_key(fast) == S.state_key(S.fast_env())


def test_patches_contract(monkeypatch):
    assert tau2_patches.PATCHES and all(isinstance(p, str) for p in tau2_patches.PATCHES)
    tau2_patches.apply()  # idempotent
    tau2_patches.apply()
    monkeypatch.delenv("TAU2_DATA_DIR")
    with pytest.raises(RuntimeError, match="TAU2_DATA_DIR"):
        tau2_patches.apply()


def test_restore_idempotent_and_non_aliasing(train_tasks):
    bad = next(t for i, t in train_tasks.items() if any(a["func_name"] == "break_vpn" for a in t["initial_state"]["initialization_actions"]))
    env = init_env(bad)
    rec = Recorder(env)
    rec.run_message([("assistant", "refuel_data", {"customer_id": "C1002", "line_id": "L1004", "gb_amount": 2.0})])
    rec.run_message([("user", "toggle_wifi", {})])
    snap = S.snapshot(env, step=2, messages=[m.model_dump() for m in rec.messages])
    frozen = copy.deepcopy((snap.agent_db, snap.user_db, snap.vpn_default, snap.messages))
    assert snap.key == S.state_key(env)
    assert snap.vpn_default["server_performance"] == "poor"

    e1, e2 = S.restore(snap), S.restore(snap)
    assert S.state_key(e1) == S.state_key(e2) == snap.key
    assert e1.tools.db is not e2.tools.db and e1.user_tools.db is not e1.tools.db
    # mutate e1 hard; neither the snapshot nor e2 may move
    for r, n, a in [("user", "toggle_airplane_mode", {}), ("user", "disconnect_vpn", {}),
                    ("assistant", "refuel_data", {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2.0}),
                    ("user", "toggle_data", {})]:
        e1.get_response(ToolCall(id="x", name=n, arguments=a, requestor=r))
    e1.tools.db.customers[0].bill_ids.append("Bzzz")
    e1.user_tools.default_vpn_details.protocol = "mutated"
    assert S.state_key(e1) != snap.key
    assert (snap.agent_db, snap.user_db, snap.vpn_default, snap.messages) == frozen
    assert S.state_key(e2) == snap.key
    assert S.state_key(S.restore(snap)) == snap.key

    # branch: independent envs
    b = S.branch(snap, 3)
    b[0].get_response(ToolCall(id="y", name="toggle_airplane_mode", arguments={}, requestor="user"))
    assert S.state_key(b[1]) == S.state_key(b[2]) == snap.key and S.state_key(b[0]) != snap.key

    # VPN alias is re-established: break_vpn after restore matches break_vpn on the live env
    live = S.fast_env()
    live.get_response(ToolCall(id="v", name="connect_vpn", arguments={}, requestor="user"))
    s2 = S.snapshot(live)
    live.user_tools.break_vpn()
    again = S.restore(s2)
    again.user_tools.break_vpn()
    assert S.state_key(again) == S.state_key(live)
    assert S.state_key(S.restore(s2)) == s2.key  # and the earlier restore did not leak into the default


def test_snapshot_id_and_parent():
    env = S.fast_env()
    a = S.snapshot(env, step=0)
    b = S.snapshot(env, parent_id=a.snapshot_id, step=1)
    assert a.snapshot_id != b.snapshot_id and b.parent_id == a.snapshot_id
    assert S.snapshot(env, step=0).snapshot_id == a.snapshot_id


def test_replay_equality_random_prefixes(train_tasks):
    rng = random.Random(0)
    ids = sorted(train_tasks)
    bad_ids = [i for i in ids if any(a["func_name"] == "break_vpn" for a in train_tasks[i]["initial_state"]["initialization_actions"])]
    assert bad_ids, "no bad_vpn task in train"
    lengths: list[int] = []
    n_write = n_total = n_err = n_hall = n_multi = 0
    intents: Counter = Counter()
    n_bad = n_pay = n_refuel = 0
    for p in range(100):
        tid = rng.choice(bad_ids) if p % 4 == 3 else rng.choice(ids)
        task = train_tasks[tid]
        intents[intent_of(tid)] += 1
        n_bad += tid in bad_ids
        length = rng.randint(3, 25)
        env = init_env(task)
        plan = gen_messages(rng, env, length)
        k = rng.randint(0, length)
        rec = Recorder(env)
        box: dict[str, Any] = {}

        def grab(n_done: int) -> None:
            if n_done == k:
                box["snap"] = S.snapshot(env, step=n_done)

        if k == 0:
            box["snap"] = S.snapshot(env, step=0)
        for msg in plan:
            rec.run_message(msg, on_call=grab)
        assert rec.n_calls == length
        live_key = S.state_key(env)

        flat = [c for m in plan for c in m]
        kit_w = {n for n in env.tools.tools if env.tools.tool_mutates_state(n)} | {n for n in env.user_tools.tools if env.user_tools.tool_mutates_state(n)}
        lengths.append(length)
        n_total += length
        n_write += sum(c[1] in kit_w for c in flat)
        n_hall += sum(c[1] in ("fix_everything", "toggle_bluetooth") for c in flat)
        n_err += sum(m.error for m in rec.tool_msgs)
        n_multi += sum(len(m) > 1 for m in plan)
        n_pay += sum(c[1] == "make_payment" for c in flat)
        n_refuel += sum(c[1] == "refuel_data" for c in flat)

        # (b) restore at k, replay the remaining calls through get_response
        renv = S.restore(box["snap"])
        assert S.state_key(renv) == box["snap"].key
        for j, (req, n, a) in enumerate(flat[k:], start=k):
            tm = renv.get_response(ToolCall(id=f"c{j}", name=n, arguments=copy.deepcopy(a), requestor=req))
            assert tm.content == rec.tool_msgs[j].content, (tid, j, n)
        assert S.state_key(renv) == live_key, (p, tid, "restore+replay")

        # (c) fresh env + set_state replay of the recorded messages
        senv = S.fast_env()
        senv.set_state(initialization_data=None, initialization_actions=init_calls(task), message_history=rec.messages)
        assert S.state_key(senv) == live_key, (p, tid, "set_state")

    stats = {
        "prefixes": 100, "calls": n_total,
        "length_hist": dict(sorted(Counter(lengths).items())),
        "write_share": round(n_write / n_total, 3), "error_tool_messages": n_err, "hallucinated_calls": n_hall,
        "multi_call_messages": n_multi, "make_payment_calls": n_pay, "refuel_calls": n_refuel,
        "intents": dict(intents), "bad_vpn_prefixes": n_bad,
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "replay_stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    print("\nREPLAY_STATS", json.dumps(stats))
    assert set(intents) == {"service_issue", "mobile_data_issue", "mms_issue"}
    assert n_bad >= 20 and n_pay > 0 and n_refuel > 0 and n_err > 0 and n_hall > 0 and n_multi > 0
    assert 0.15 < n_write / n_total < 0.9


_VPN_SCRIPT = """
import json, sys
from telecom_boed import paths
mode, patch, spec = sys.argv[1], sys.argv[2], sys.argv[3]
if patch == "yes":
    from telecom_boed.env import tau2_patches
    tau2_patches.apply()
from telecom_boed.env import snapshot as S
from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
tasks = json.load(open(spec))
out = []
for actions in tasks:
    if mode == "break":
        env = S.fast_env()
        for a in actions:
            env.run_env_function_call(EnvFunctionCall(**a))
    fresh = S.fast_env()
    res = [fresh.get_response(ToolCall(id=str(i), name=n, arguments={}, requestor="user")).content for i, n in enumerate(["connect_vpn", "run_speed_test"])]
    out.append([res, S.state_key(fresh)[2]])
print("RESULT" + json.dumps(out))
"""


def _run_vpn(tmp_path: Path, mode: str, patch: str, spec: Path) -> list:
    script = tmp_path / "vpn_probe.py"
    script.write_text(_VPN_SCRIPT)
    env = dict(os.environ)
    p = subprocess.run([sys.executable, "-I", str(script), mode, patch, str(spec)], capture_output=True, text=True,
                       cwd=tmp_path, env={**env, "PYTHONPATH": str(paths.PACKAGE_ROOT / "src")}, timeout=600)
    assert p.returncode == 0, p.stderr[-1500:]
    line = next(l for l in p.stdout.splitlines() if l.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def test_vpn_isolation_subprocess(train_tasks, tmp_path):
    bad = [t for t in train_tasks.values() if any(a["func_name"] == "break_vpn" for a in t["initial_state"]["initialization_actions"])]
    assert len(bad) >= 5
    spec = tmp_path / "bad_vpn_actions.json"
    spec.write_text(json.dumps([t["initial_state"]["initialization_actions"] for t in bad]))
    broken_p = _run_vpn(tmp_path, "break", "yes", spec)
    clean_p = _run_vpn(tmp_path, "clean", "yes", spec)
    assert broken_p == clean_p  # patched: break_vpn never leaks into later envs
    assert len({json.dumps(r) for r in clean_p}) == 1
    broken_u = _run_vpn(tmp_path, "break", "no", spec)
    clean_u = _run_vpn(tmp_path, "clean", "no", spec)
    assert broken_u != clean_u, "unpatched leak not reproduced; the isolation test would be meaningless"
    assert clean_u == clean_p  # a clean process gives the same answer with or without the patch


def test_deterministic_bill_ids():
    plans = {
        "C1002 no draft": [("assistant", "refuel_data", {"customer_id": "C1002", "line_id": "L1004", "gb_amount": 2.0}),
                           ("user", "toggle_wifi", {}),
                           ("assistant", "refuel_data", {"customer_id": "C1002", "line_id": "L1005", "gb_amount": 1.0})],
        "C1003 no draft": [("assistant", "refuel_data", {"customer_id": "C1003", "line_id": "L1009", "gb_amount": 2.0})],
        "C1001 draft exists": [("assistant", "refuel_data", {"customer_id": "C1001", "line_id": "L1002", "gb_amount": 2.0})],
    }
    ids = {}
    for name, flat in plans.items():
        env_a = S.fast_env()
        n0 = len(env_a.tools.db.bills)
        rec = Recorder(env_a)
        snaps = [S.snapshot(env_a)]
        for c in flat:
            rec.run_message([c])
            snaps.append(S.snapshot(env_a, step=len(snaps)))
        keys = S.state_key(env_a)
        if "no draft" in name and "C1003" not in name:
            assert len(env_a.tools.db.bills) == n0 + 1, "expected a freshly minted bill"
        ids[name] = [b.bill_id for b in env_a.tools.db.bills[n0:]]
        # (a) a second live run in the same process, after other envs minted ids
        env_b = S.fast_env()
        for c in flat:
            Recorder(env_b).run_message([c])
        assert S.state_key(env_b) == keys, name
        # (b) restore from the snapshot before the first call and from the middle, then replay the rest
        for k in range(len(flat) + 1):
            renv = S.restore(snaps[k])
            for j, (req, n, a) in enumerate(flat[k:], start=k):
                renv.get_response(ToolCall(id=f"r{j}", name=n, arguments=a, requestor=req))
            assert S.state_key(renv) == keys, (name, k)
        # (c) set_state replay
        senv = S.fast_env()
        senv.set_state(initialization_data=None, initialization_actions=[], message_history=rec.messages)
        assert S.state_key(senv) == keys, name
    assert len(ids["C1002 no draft"]) == 1 and ids["C1001 draft exists"] == []
    expected = "B" + hashlib.sha256(f"C1002|{len(S.fast_env().tools.db.bills)}".encode()).hexdigest()[:8]
    assert ids["C1002 no draft"] == [expected]  # content-derived: customer id + number of bills


def test_timings_written(train_tasks):
    task = next(iter(train_tasks.values()))
    reps = 300
    base = init_env(task)
    snap = S.snapshot(base)
    tc = [ToolCall(id="t", name="toggle_wifi", arguments={}, requestor="user")]

    def bench(fn) -> dict:
        xs = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t0) * 1e6)
        xs.sort()
        return {"mean_us": round(statistics.fmean(xs), 1), "p50_us": round(xs[len(xs) // 2], 1),
                "p95_us": round(xs[int(len(xs) * 0.95)], 1), "reps": reps}

    out = {
        "fast_env": bench(S.fast_env),
        "snapshot": bench(lambda: S.snapshot(base)),
        "restore": bench(lambda: S.restore(snap)),
        "get_response_step": bench(lambda: base.get_response(tc[0])),
        "state_key": bench(lambda: S.state_key(base)),
        "get_environment_reference": bench(get_environment),
    }
    TIMINGS.parent.mkdir(parents=True, exist_ok=True)
    TIMINGS.write_text(json.dumps(out, indent=1) + "\n")
    print("\nTIMINGS", json.dumps(out))
    assert out["restore"]["p50_us"] < out["get_environment_reference"]["p50_us"]
