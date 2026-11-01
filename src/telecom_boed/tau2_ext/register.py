"""[telecom_boed] Named run configs + registration of our agents/users into tau2's registry (W3-S10).

Run config (configs/runs/<name>.yaml; immutable once used, add a new name instead of editing). Unknown keys are errors.
    name           str, == file stem               status       ready | template   (template = names not registered yet)
    agent          llm_agent | boed-plain | boed-controller | any registered name   (llm_agent_gt/_solo are refused:
                   they read the task)
    agent_kwargs   dict, only for non-native agents (e.g. {objective: reach})
    user           user_simulator (native tau2 user) | boed-user-<persona> | any registered name
    domain         telecom | telecom-workflow      split        train | test | base | small | full
    num_trials, seed, max_steps, max_concurrency, max_errors     ints (defaults = tau2 CLI defaults: 1, 300, 200, 3, 10)
    task_ids       optional list[str]              num_tasks    optional int (not both)
    llm            {model, temperature, extra_body}   (agent; api_base from env VLLM_API_BASE, api_key "local")
    user_llm       optional, same shape; defaults to `llm`
    boed_config    path to a configs/*.yaml knob file (default configs/boed_default.yaml)
    eval_final     bool; REQUIRED true for any split that contains test-40 (test, base, small, full)

How our kwargs reach our factories without touching `task`: for non-native agents/users the launcher adds one extra key
`_boed` to llm_args_agent / llm_args_user = {run, agent_kwargs|user, boed_config, boed_config_sha256}. tau2 hands llm_args
to the factory/constructor untouched (runner/build.py:117-123, 170-173); our factories call `split_boed_args(llm_args)`
to strip it before anything reaches litellm. The same data is available by name from `get_run(run_name)` (module registry).
"""
from __future__ import annotations

import hashlib
import importlib
import logging
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from telecom_boed import paths

log = logging.getLogger(__name__)

BOED_KEY = "_boed"
NATIVE_AGENTS = frozenset({"llm_agent"})
NATIVE_USERS = frozenset({"user_simulator"})
FORBIDDEN_AGENTS = frozenset({"llm_agent_gt", "llm_agent_solo", "discrete_time_audio_native_agent"})
FORBIDDEN_USERS = frozenset({"dummy_user", "voice_streaming_user_simulator"})
DOMAINS = ("telecom", "telecom-workflow")
SPLITS = ("train", "test", "base", "small", "full")
TEST_BEARING_SPLITS = frozenset({"test", "base", "small", "full"})  # base = train + test-40


def train_safe(task_ids) -> bool:
    """True iff no id is in test-40 and no id shares a test-40 FAULT SET (any intent): the branch-pool rule of
    testing/leakage_scan.branch_pool_task_ids. Lets training-data runs use the full catalog without eval_final."""
    import json as _json
    from telecom_boed import paths as _paths
    _, split_path = _paths.telecom_catalog_paths()
    test = set(_json.loads(split_path.read_text())["test"])
    fs = lambda t: t.split("]", 1)[1].split("[", 1)[0]
    test_fs = {fs(t) for t in test}
    return all(t not in test and fs(t) not in test_fs for t in task_ids)
ORACLE_AGENTS = frozenset({"oracle-teacher"})  # privileged scripted teacher (rollout/oracle_agent.py): train-safe tasks only
PLUGIN_MODULES = ("telecom_boed.tau2_ext.agents", "telecom_boed.tau2_ext.personas",
                  "telecom_boed.rollout.oracle_agent", "telecom_boed.tau2_ext.belief_script")  # each: register(registry) -> names
DEFAULT_BOED_CONFIG = "configs/boed_default.yaml"
RUNS_DIR = paths.CONFIG_DIR / "runs"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class ConfigError(ValueError):
    """Invalid run config."""


def check_oracle_allowed(agent: str, split: str, eval_final: bool, task_ids=None) -> None:
    """Refuse the privileged teacher (ORACLE_AGENTS) under eval_final, or unless every task it could run is train-safe
    (explicit task_ids, else the whole `split` from the shipped split file). No-op for other agents."""
    if agent not in ORACLE_AGENTS:
        return
    if eval_final:
        raise ConfigError(f"agent {agent!r} reads the hidden task and is refused with eval_final: true")
    if task_ids:
        ids = list(task_ids)
    else:
        import json as _json
        _, split_path = paths.telecom_catalog_paths()
        ids = _json.loads(split_path.read_text()).get(split, [])
        if split in TEST_BEARING_SPLITS or not ids:
            raise ConfigError(f"agent {agent!r} needs train-safe tasks: split {split!r} contains test-40 "
                              "(give train-safe task_ids or use split: train)")
    if not train_safe(ids):
        raise ConfigError(f"agent {agent!r} is refused: task ids are not train-safe (test-40 id or test-40 fault set)")


@dataclass(frozen=True)
class LLMSpec:
    model: str
    temperature: float = 0.0
    extra_body: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    name: str
    llm: LLMSpec
    agent: str = "llm_agent"
    agent_kwargs: dict = field(default_factory=dict)
    user: str = "user_simulator"
    user_llm: LLMSpec | None = None
    domain: str = "telecom"
    split: str = "train"
    num_trials: int = 1
    seed: int = 300
    max_steps: int = 200
    max_concurrency: int = 3
    max_errors: int = 10
    task_ids: tuple[str, ...] | None = None
    num_tasks: int | None = None
    boed_config: str = DEFAULT_BOED_CONFIG
    eval_final: bool = False
    status: str = "ready"

    def to_dict(self) -> dict:
        return asdict(self)


_LLM_KEYS = {"model", "temperature", "extra_body"}
_KEYS = {f for f in RunSpec.__dataclass_fields__}


def _int(d: dict, key: str, lo: int, default: int) -> int:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ConfigError(f"{key}: expected int >= {lo}, got {v!r}")
    return v


def _llm(raw: Any, where: str) -> LLMSpec:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping {{model, temperature, extra_body}}")
    bad = set(raw) - _LLM_KEYS
    if bad:
        raise ConfigError(f"{where}: unknown keys {sorted(bad)}")
    if not isinstance(raw.get("model"), str) or not raw["model"]:
        raise ConfigError(f"{where}.model: required non-empty string")
    t = raw.get("temperature", 0.0)
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
        raise ConfigError(f"{where}.temperature: expected number >= 0")
    eb = raw.get("extra_body", {})
    if not isinstance(eb, dict):
        raise ConfigError(f"{where}.extra_body: expected a mapping")
    return LLMSpec(raw["model"], float(t), dict(eb))


def parse_run_config(raw: Any, stem: str | None = None) -> RunSpec:
    """Strictly validate a run-config mapping. `stem` (file stem) must equal `name` when given."""
    if not isinstance(raw, dict):
        raise ConfigError("run config must be a mapping")
    if "workers" in raw:
        raise ConfigError("workers is not configurable: runs are in-process (workers=0)")
    bad = set(raw) - _KEYS
    if bad:
        raise ConfigError(f"unknown keys {sorted(bad)}")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ConfigError(f"name: required, [A-Za-z0-9_.-]+, got {name!r}")
    if stem is not None and stem != name:
        raise ConfigError(f"name {name!r} != file stem {stem!r}")
    if "llm" not in raw:
        raise ConfigError("llm: required")
    llm = _llm(raw["llm"], "llm")
    user_llm = _llm(raw["user_llm"], "user_llm") if raw.get("user_llm") is not None else None
    agent, user = raw.get("agent", "llm_agent"), raw.get("user", "user_simulator")
    for key, val in (("agent", agent), ("user", user)):
        if not isinstance(val, str) or not val:
            raise ConfigError(f"{key}: expected non-empty string")
    if agent in FORBIDDEN_AGENTS:
        raise ConfigError(f"agent {agent!r} is refused (reads the task / not a text agent)")
    if user in FORBIDDEN_USERS:
        raise ConfigError(f"user {user!r} is refused")
    kw = raw.get("agent_kwargs", {})
    if not isinstance(kw, dict) or not all(isinstance(k, str) for k in kw):
        raise ConfigError("agent_kwargs: expected a mapping with string keys")
    if kw and agent in NATIVE_AGENTS:
        raise ConfigError(f"agent_kwargs is not allowed for native agent {agent!r}")
    domain, split = raw.get("domain", "telecom"), raw.get("split", "train")
    if domain not in DOMAINS:
        raise ConfigError(f"domain: one of {DOMAINS}, got {domain!r}")
    if split not in SPLITS:
        raise ConfigError(f"split: one of {SPLITS}, got {split!r}")
    ef = raw.get("eval_final", False)
    if not isinstance(ef, bool):
        raise ConfigError("eval_final: expected bool")
    ids = raw.get("task_ids")
    if ids is not None and (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)):
        raise ConfigError("task_ids: expected a non-empty list of strings")
    if split in TEST_BEARING_SPLITS and not ef and not (ids and train_safe(ids)):
        raise ConfigError(f"split {split!r} contains test-40: eval_final: true is required "
                          "(or give task_ids that are train-safe: no test-40 id and no test-40 fault set)")
    check_oracle_allowed(agent, split, ef, ids)
    nt = raw.get("num_tasks")
    if nt is not None and (isinstance(nt, bool) or not isinstance(nt, int) or nt < 1):
        raise ConfigError("num_tasks: expected int >= 1")
    if ids is not None and nt is not None:
        raise ConfigError("task_ids and num_tasks are mutually exclusive")
    bc = raw.get("boed_config", DEFAULT_BOED_CONFIG)
    if not isinstance(bc, str) or not resolve_boed_config(bc).is_file():
        raise ConfigError(f"boed_config: file not found: {bc!r}")
    status = raw.get("status", "ready")
    if status not in ("ready", "template"):
        raise ConfigError("status: ready | template")
    return RunSpec(
        name=name, llm=llm, agent=agent, agent_kwargs=dict(kw), user=user, user_llm=user_llm, domain=domain,
        split=split, num_trials=_int(raw, "num_trials", 1, 1), seed=_int(raw, "seed", 0, 300),
        max_steps=_int(raw, "max_steps", 1, 200), max_concurrency=_int(raw, "max_concurrency", 1, 3),
        max_errors=_int(raw, "max_errors", 1, 10), task_ids=tuple(ids) if ids else None, num_tasks=nt,
        boed_config=bc, eval_final=ef, status=status)


def resolve_boed_config(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else paths.PACKAGE_ROOT / q


def load_run_config(path: str | Path) -> tuple[RunSpec, str]:
    """(spec, raw file text). The text is what run_meta hashes."""
    p = Path(path)
    if not p.is_absolute() and not p.exists() and (RUNS_DIR / p).exists():
        p = RUNS_DIR / p
    text = p.read_text()
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"{p}: invalid YAML: {e}") from e
    return parse_run_config(raw, stem=p.stem), text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------------------------------- side channel
_RUNS: dict[str, RunSpec] = {}


def set_run(spec: RunSpec) -> None:
    _RUNS[spec.name] = spec


def get_run(name: str) -> RunSpec:
    return _RUNS[name]


def boed_payload(spec: RunSpec, role: str) -> dict:
    """The `_boed` dict injected into llm_args_agent (role="agent") / llm_args_user (role="user")."""
    bc = resolve_boed_config(spec.boed_config)
    out = {"run": spec.name, "boed_config": str(bc), "boed_config_sha256": sha256_text(bc.read_text())}
    if role == "agent":
        out["agent_kwargs"] = dict(spec.agent_kwargs)
    else:
        out["user"] = spec.user
    return out


def split_boed_args(llm_args: dict | None) -> tuple[dict, dict]:
    """(llm_args without `_boed`, the `_boed` payload or {}). Our factories call this first; llm_args is not mutated."""
    la = dict(llm_args or {})
    return la, dict(la.pop(BOED_KEY, None) or {})


def llm_args_for(spec: RunSpec, role: str, env: dict[str, str] | None = None) -> dict:
    """litellm kwargs (+ `_boed` for non-native participants). api_base from VLLM_API_BASE (hosted_vllm/ needs it)."""
    import os

    env = os.environ if env is None else env
    l = spec.llm if role == "agent" or spec.user_llm is None else spec.user_llm
    args: dict = {"temperature": l.temperature}
    if env.get("VLLM_API_BASE"):
        args.update(api_base=env["VLLM_API_BASE"], api_key="local")
    elif l.model.startswith("hosted_vllm/"):
        raise ConfigError("VLLM_API_BASE is unset but the model is hosted_vllm/")
    if l.extra_body:
        args["extra_body"] = dict(l.extra_body)
    native = spec.agent in NATIVE_AGENTS if role == "agent" else spec.user in NATIVE_USERS
    if not native:
        args[BOED_KEY] = boed_payload(spec, role)
    return args


# ------------------------------------------------------------------------------------------- registration
class _IdempotentRegistry:
    """Proxy over tau2's Registry: re-registering the *same* object under a name is a no-op (tau2 raises ValueError);
    a different object under an existing name still raises. Records the names it registered."""

    def __init__(self, reg: Any):
        self._reg, self.users, self.agents = reg, [], []

    def register_user(self, user_constructor: type, name: str | None = None) -> None:
        key = name or user_constructor.__name__
        if key in self._reg.get_users():
            if self._reg.get_user_constructor(key) is user_constructor:
                return
        else:
            self.users.append(key)
        self._reg.register_user(user_constructor, name)

    def register_agent_factory(self, factory: Any, name: str, **kw: Any) -> None:
        if name in self._reg.get_agents():
            if self._reg.get_agent_factory(name) is factory:
                return
        else:
            self.agents.append(name)
        self._reg.register_agent_factory(factory, name, **kw)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._reg, item)


_REGISTERED: list[str] | None = None


def register_all() -> list[str]:
    """Register our agents/users (S11 agents.py, S12 personas.py) once per process; returns the names registered.
    A missing plugin module is skipped with a notice; any other error (incl. a broken import inside it) raises."""
    global _REGISTERED
    if _REGISTERED is not None:
        return list(_REGISTERED)
    from tau2.registry import registry

    proxy = _IdempotentRegistry(registry)
    names: list[str] = []
    for mod_name in PLUGIN_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name != mod_name:
                raise
            msg = f"[telecom_boed] register: {mod_name} not present yet, skipped"
            log.warning(msg)
            print(msg, file=sys.stderr)
            continue
        names += list(mod.register(proxy))
    _REGISTERED = sorted(set(names))
    return list(_REGISTERED)


def registered_names() -> dict[str, list[str]]:
    from tau2.registry import registry

    return {"agents": sorted(registry.get_agents()), "users": sorted(registry.get_users())}


def check_names(spec: RunSpec) -> None:
    """Fail loudly if the config's agent/user is not registered (call after register_all)."""
    check_oracle_allowed(spec.agent, spec.split, spec.eval_final, spec.task_ids)
    have = registered_names()
    if spec.agent not in have["agents"]:
        raise ConfigError(f"agent {spec.agent!r} is not registered (have {have['agents']})")
    if spec.user not in have["users"]:
        raise ConfigError(f"user {spec.user!r} is not registered (have {have['users']})")
