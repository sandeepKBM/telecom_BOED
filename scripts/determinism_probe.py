#!/usr/bin/env python3
"""Determinism probe for tau2 user-simulator calls against a local vLLM (W3-D1, 2026-10-16).

Two modes (run with the ddl-telecom python, which has tau2 + litellm):

  capture --out REQ.json
      Build the REAL first user-simulator request for three TRAIN telecom tasks via tau2's own
      build_environment/build_user (system prompt = global guidelines + scenario; tools = user tools;
      one prior assistant message "Hi! How can I help you today?"), by intercepting tau2's `generate`.
      Needs TAU2_DATA_DIR (read-only use). CPU only.

  matrix --req REQ.json --out OUT.json [--configs a,b,c] [--launches 2]
      For each vLLM config, start a server (own process group, GPU = CUDA_VISIBLE_DEVICES from caller),
      run the battery, kill it, repeat `--launches` times (fresh server each).  Battery per launch:
        R x5 sequential (same request)            -> within-server repeatability
        O (shared-prefix sibling), R, P (other), R  -> (d) a different request in between
        R via litellm (hosted_vllm/Qwen/Qwen3-8B)  -> client-difference check
        decode throughput: 1 stream and 8 concurrent streams, 256 forced tokens
      Raw requests use urllib with logprobs/top_logprobs=2 (litellm one has no logprobs).
  configs: a=default flags, b=--no-enable-prefix-caching, c=VLLM_BATCH_INVARIANT=1 + --enforce-eager + no prefix cache.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request, threading, shutil

MODEL = "Qwen/Qwen3-8B"
SEED = 626729
GREETING = "Hi! How can I help you today?"
TASKS = [  # train tasks from det_llmage_526576.24 (index 1 diverged between det_* runs; 0 shares its prefix)
    "[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]",
    "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]",
    "[mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]",
]


# ----------------------------------------------------------------------------- capture
def capture(out):
    from tau2.run import get_tasks
    from tau2.runner.build import build_environment, build_user
    from tau2.data_model.message import AssistantMessage
    import tau2.user.user_simulator as us

    class Stop(Exception):
        pass

    got = {}

    def fake_generate(model, messages, tools=None, tool_choice=None, **kw):
        from tau2.utils.llm_utils import to_litellm_messages
        got["messages"] = to_litellm_messages(messages)
        got["tools"] = [t.openai_schema for t in tools] if tools else None
        got["kw"] = kw
        raise Stop

    us.generate = fake_generate
    tasks = {t.id: t for t in get_tasks("telecom", task_split_name="train")}
    llm_args = {"temperature": 0.0, "api_base": "http://127.0.0.1:PORT/v1", "api_key": "local",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    reqs = []
    for tid in TASKS:
        task = tasks[tid]
        env = build_environment("telecom")
        user = build_user("user_simulator", env, task, llm="hosted_vllm/" + MODEL, llm_args=dict(llm_args))
        user.set_seed(SEED)
        st = user.get_init_state()
        got.clear()
        try:
            user.generate_next_message(AssistantMessage(role="assistant", content=GREETING), st)
        except Stop:
            pass
        kw = {k: v for k, v in got["kw"].items() if k != "call_name"}
        reqs.append({"task_id": tid, "messages": got["messages"], "tools": got["tools"],
                     "tool_choice": "auto" if got["tools"] else None, "litellm_kwargs": kw})
    json.dump(reqs, open(out, "w"), indent=1)
    for r in reqs:
        print(r["task_id"][:50], "n_msgs", len(r["messages"]), "sys_chars", len(r["messages"][0]["content"]),
              "n_tools", len(r["tools"] or []), "kw", sorted(r["litellm_kwargs"]))


# ----------------------------------------------------------------------------- client helpers
def post(port, body, timeout=600):
    rq = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json", "Authorization": "Bearer local"})
    return json.loads(urllib.request.urlopen(rq, timeout=timeout).read())


def raw_body(r, logprobs=True, **extra):
    b = {"model": MODEL, "messages": r["messages"], "temperature": 0.0, "seed": SEED, "max_tokens": 512,
         "chat_template_kwargs": {"enable_thinking": False}}
    if r["tools"]:
        b["tools"] = r["tools"]; b["tool_choice"] = "auto"
    if logprobs:
        b["logprobs"] = True; b["top_logprobs"] = 2
    b.update(extra)
    return b


def summarize(resp):
    ch = resp["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    toks = []
    for e in lp:
        top = sorted(e.get("top_logprobs") or [], key=lambda x: -x["logprob"])[:2]
        toks.append([e["token"], e["logprob"], [(t["token"], t["logprob"]) for t in top]])
    msg = ch["message"]
    text = msg.get("content") or ""
    if msg.get("tool_calls"):
        text += " <TOOLCALLS> " + json.dumps([(c["function"]["name"], c["function"]["arguments"]) for c in msg["tool_calls"]])
    return {"text": text, "toks": toks, "finish": ch.get("finish_reason"),
            "n_prompt": resp["usage"]["prompt_tokens"], "n_out": resp["usage"]["completion_tokens"]}


def first_div(a, b):
    """token position of first divergence between two summarized outputs, with top-2 gap at that position (in a and b)."""
    ta, tb = [t[0] for t in a["toks"]], [t[0] for t in b["toks"]]
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            ga = a["toks"][i][2]; gb = b["toks"][i][2]
            gap = lambda t: (t[0][1] - t[1][1]) if len(t) > 1 else None
            return {"pos": i, "tok_a": ta[i], "tok_b": tb[i], "gap_a": gap(ga), "gap_b": gap(gb),
                    "lp_a": a["toks"][i][1], "lp_b": b["toks"][i][1]}
    if len(ta) != len(tb):
        return {"pos": min(len(ta), len(tb)), "note": "length differs"}
    return None


def logprob_drift(a, b):
    """max |logprob diff| over the common identical-token prefix (nonzero => numerics differ even if text equal)."""
    m = 0.0
    for x, y in zip(a["toks"], b["toks"]):
        if x[0] != y[0]:
            break
        m = max(m, abs(x[1] - y[1]))
    return m


# ----------------------------------------------------------------------------- battery
def battery(port, reqs, vcfg_note):
    R, O, P = reqs[1], reqs[0], reqs[2]
    res = {}
    t = time.time()
    res["R_x5"] = [summarize(post(port, raw_body(R))) for _ in range(5)]
    res["R_x5_s"] = time.time() - t
    res["seq_d"] = {}
    o1 = summarize(post(port, raw_body(O)))
    r_after_o = summarize(post(port, raw_body(R)))
    p1 = summarize(post(port, raw_body(P)))
    r_after_p = summarize(post(port, raw_body(R)))
    res["d_O"], res["d_R_after_O"], res["d_P"], res["d_R_after_P"] = o1, r_after_o, p1, r_after_p
    # litellm
    try:
        import litellm
        kw = dict(R["litellm_kwargs"]); kw["api_base"] = f"http://127.0.0.1:{port}/v1"
        r = litellm.completion(model="hosted_vllm/" + MODEL, messages=R["messages"], tools=R["tools"],
                               tool_choice=R["tool_choice"], num_retries=0, **kw)
        ch = r.choices[0]; text = ch.message.content or ""
        if ch.message.tool_calls:
            text += " <TOOLCALLS> " + json.dumps([(c.function.name, c.function.arguments) for c in ch.message.tool_calls])
        res["litellm"] = {"text": text, "kw_keys": sorted(kw)}
    except Exception as e:  # noqa
        res["litellm"] = {"error": repr(e)[:300]}
    # throughput: forced 256 tokens
    def one():
        b = raw_body(R, logprobs=False, max_tokens=256, min_tokens=256, ignore_eos=True)
        return post(port, b)["usage"]["completion_tokens"]
    t = time.time(); n = one(); res["tps_1stream"] = n / (time.time() - t)
    outs = []
    th = [threading.Thread(target=lambda: outs.append(one())) for _ in range(8)]
    t = time.time(); [x.start() for x in th]; [x.join() for x in th]
    res["tps_8stream"] = sum(outs) / (time.time() - t)
    return res


CONFIGS = {
    "a": {"env": {}, "args": ["--enable-prefix-caching"], "desc": "default flags (prefix cache on, cudagraph/compile)"},
    "b": {"env": {}, "args": ["--no-enable-prefix-caching"], "desc": "--no-enable-prefix-caching"},
    "c": {"env": {"VLLM_BATCH_INVARIANT": "1"}, "args": ["--no-enable-prefix-caching", "--enforce-eager"],
          "desc": "VLLM_BATCH_INVARIANT=1 + --enforce-eager + no prefix cache"},
}


def launch(cfg, port, logf, cache_root):
    env = dict(os.environ); env.update(CONFIGS[cfg]["env"]); env["VLLM_CACHE_ROOT"] = cache_root
    cmd = [os.environ["VLLM_ENV"] + "/bin/vllm", "serve", MODEL, "--host", "127.0.0.1", "--port", str(port),
           "--served-model-name", MODEL, "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--seed", "0"] + CONFIGS[cfg]["args"]
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, stdout=open(logf, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise RuntimeError("vllm died; see " + logf)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p, time.time() - t0
        except Exception:
            time.sleep(3)
    raise RuntimeError("vllm not ready")


def kill(p):
    try:
        os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=60)
    except Exception:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    time.sleep(8)


def matrix(a):
    reqs = json.load(open(a.req))
    out = {"configs": {}}
    port = int(os.environ.get("PROBE_PORT", "8777"))
    workdir = os.environ["TMPDIR"]
    for cfg in a.configs.split(","):
        out["configs"][cfg] = {"desc": CONFIGS[cfg]["desc"], "launches": []}
        for L in range(a.launches):
            cache = f"{workdir}/vc_{cfg}_{L}"; shutil.rmtree(cache, ignore_errors=True)  # cold compile cache for every launch (uniform)
            p, ready = launch(cfg, port, f"{workdir}/vllm_{cfg}_{L}.log", cache)
            try:
                b = battery(port, reqs, cfg); b["ready_s"] = ready
            finally:
                kill(p)
            out["configs"][cfg]["launches"].append(b)
            json.dump(out, open(a.out, "w"))
            print(f"[{cfg} launch {L}] ready {ready:.0f}s tps1 {b['tps_1stream']:.1f} tps8 {b['tps_8stream']:.1f}", flush=True)
    return out


def analyze(path):
    d = json.load(open(path)); rows = []
    for cfg, c in d["configs"].items():
        for L, b in enumerate(c["launches"]):
            R = b["R_x5"]
            fd = [first_div(R[0], x) for x in R[1:]]
            rows.append(f"{cfg} L{L} R x5 identical={all(x is None for x in fd)} first-div={[f for f in fd if f][:1]} "
                        f"maxlogprob-drift={max(logprob_drift(R[0], x) for x in R[1:]):.2e}")
            for k in ("d_R_after_O", "d_R_after_P"):
                fdv = first_div(R[0], b[k])
                rows.append(f"  R0 vs {k}: identical={fdv is None} {fdv or ''} drift={logprob_drift(R[0], b[k]):.2e}")
            rows.append(f"  drift R1..R4 vs R1 (all cache-hit): {max(logprob_drift(R[1], x) for x in R[2:]):.2e}; "
                        f"R0 prompt tokens {R[0]['n_prompt']} out {R[0]['n_out']}")
            rows.append(f"  litellm text == R0 text: {b['litellm'].get('text') == R[0]['text']}")
        ls = c["launches"]
        if len(ls) > 1:
            fdv = first_div(ls[0]["R_x5"][0], ls[1]["R_x5"][0])
            rows.append(f"{cfg} R0 across relaunch: identical={fdv is None} {fdv or ''}")
    # cross-config / cross-launch comparison of the R0 text
    base = d["configs"][next(iter(d["configs"]))]["launches"][0]["R_x5"][0]
    for cfg, c in d["configs"].items():
        for L, b in enumerate(c["launches"]):
            fdv = first_div(base, b["R_x5"][0])
            rows.append(f"R0[{cfg}{L}] vs R0[first]: identical={fdv is None} {fdv or ''}")
    for cfg, c in d["configs"].items():
        ls = c["launches"]
        rows.append(f"{cfg}: ready_s={[round(x['ready_s']) for x in ls]} tps1={[round(x['tps_1stream'],1) for x in ls]} tps8={[round(x['tps_8stream']) for x in ls]}")
    print("\n".join(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    c = sp.add_parser("capture"); c.add_argument("--out", required=True)
    m = sp.add_parser("matrix"); m.add_argument("--req", required=True); m.add_argument("--out", required=True)
    m.add_argument("--configs", default="a,b,c"); m.add_argument("--launches", type=int, default=2)
    z = sp.add_parser("analyze"); z.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "capture": capture(a.out)
    elif a.cmd == "matrix": matrix(a); analyze(a.out)
    elif a.cmd == "analyze": analyze(a.path)
    else: ap.print_help()
