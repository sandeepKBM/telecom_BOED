"""[telecom_boed] W4-S16: study runner. A study YAML lists arms (run configs), users and a task partition; this module expands it
into launcher steps (slurm/local_tau2_vllm.sh MODE=launch) inside the shared GPU hold, each wrapped in `cdx-agent --gpu-lock <hold>`
(flock protocol + srun --overlap; the launcher sees CDX_GPU_LOCK_WRAPPER and uses the wrapper's GPU), and keeps a machine-written manifest.

Study YAML (configs/studies/<name>.yaml), strict keys:
  name, hold_job (526576), gpus ([0,1,3]; ADVISORY: `cdx-agent --gpu-lock` picks the lowest free GPU itself, so indices cannot be
  pinned; len(gpus) only bounds the width used for the wall-time estimate), max_concurrent (2), deterministic (true -> DETERMINISTIC=1),
  arms ([run-config names; native user, e.g. p4_plain_native]), users ([native | mix | <persona>]), split (train),
  task_ids (optional list; default = the whole split), tau2_timeout_s / eta_min (optional; default derived per step),
  tag_prefix ("st"), mem_gb (40), host ("rlab1"), sec_per_episode (45), startup_s (180) [the last two feed `plan` and the
  derived timeouts only].

Users: `native` = tau2's built-in user on the full task list (its own condition); `mix` = the six personas REPLACE the built-in
persona, and since a run config has ONE user the sorted task list is dealt round-robin to the six personas (one step each);
`<persona>` = one persona on the full list. Per-persona derived configs are written as NEW files configs/runs/<arm>__<persona>.yaml
(arm = native config name without its `_native` suffix) and never overwritten (a differing existing file is an error).

Manifest results/studies/<study>/manifest.jsonl (append-only, machine-written):
  {t, study, arm, user, run_config, tag, gpu, state: queued|running|done|failed, run_dir, exit_rc, attempt, n_expected, condition}
Resumable: a step is skipped when a run_dir of its tag (any attempt) holds a results.json with n_expected simulations.
No tau2 import happens here (names are checked only when `check_names=True`, which imports tau2 via register_all).
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import json
import math
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from telecom_boed import paths
from telecom_boed.tau2_ext import register

PERSONAS: tuple[str, ...] = ("cooperative", "terse", "confused", "contradictory", "noncompliant", "delayed")  # = tau2_ext.personas.PERSONAS (tested)
PERSONA_CODE = {"cooperative": "co", "terse": "te", "confused": "cf", "contradictory": "cd", "noncompliant": "nc", "delayed": "dl"}
STUDIES_DIR = paths.CONFIG_DIR / "studies"
RESULTS_STUDIES = paths.PACKAGE_ROOT / "results" / "studies"
RESULTS_RUNS = paths.PACKAGE_ROOT / "results" / "runs"
HEALTH_TIMEOUT_S = 600  # launcher default; added to the srun limit
MAX_TAG = 12  # AF_UNIX socket path limit
GPU_BUSY_RCS = (3, 4)  # cdx-agent --gpu-lock / launcher: 3 = no free GPU lock, 4 = free locks only on busy GPUs -> queued, retry later
MAX_BUSY_RETRIES = 30  # x busy_cooldown_s (120 s) = 1 h of waiting for another project's GPU before a step counts as failed
CDX_AGENT = "cdx-agent"
LAUNCHER_ABS = paths.PACKAGE_ROOT / "slurm" / "local_tau2_vllm.sh"  # the wrapper's step starts in $HOME: absolute path required
_STUDY_KEYS = {"name", "hold_job", "gpus", "max_concurrent", "deterministic", "arms", "users", "split", "task_ids", "tau2_timeout_s",
               "eta_min", "tag_prefix", "mem_gb", "host", "sec_per_episode", "startup_s"}


class StudyError(ValueError):
    """Invalid study spec."""


@dataclass(frozen=True)
class Study:
    name: str
    arms: tuple[str, ...]
    users: tuple[str, ...]
    hold_job: int = 526576
    gpus: tuple[int, ...] = (0, 1, 3)
    max_concurrent: int = 2
    deterministic: bool = True
    split: str = "train"
    task_ids: tuple[str, ...] | None = None
    tau2_timeout_s: int | None = None
    eta_min: int | None = None
    tag_prefix: str = "st"
    mem_gb: int = 40
    host: str = "rlab1"
    sec_per_episode: float = 45.0
    startup_s: float = 180.0


def _posint(raw: dict, key: str, default: int | None, lo: int = 1) -> int | None:
    v = raw.get(key, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise StudyError(f"{key}: expected int >= {lo}, got {v!r}")
    return v


def parse_study(raw: Any, stem: str | None = None) -> Study:
    if not isinstance(raw, dict):
        raise StudyError("study must be a mapping")
    bad = set(raw) - _STUDY_KEYS
    if bad:
        raise StudyError(f"unknown keys {sorted(bad)}")
    name = raw.get("name")
    if not isinstance(name, str) or not register._NAME_RE.match(name):
        raise StudyError(f"name: required, [A-Za-z0-9_.-]+, got {name!r}")
    if stem is not None and stem != name:
        raise StudyError(f"name {name!r} != file stem {stem!r}")
    arms, users = raw.get("arms"), raw.get("users")
    if not isinstance(arms, list) or not arms or not all(isinstance(a, str) and a for a in arms) or len(set(arms)) != len(arms):
        raise StudyError("arms: expected a non-empty list of distinct run-config names")
    if not isinstance(users, list) or not users or len(set(users)) != len(users):
        raise StudyError("users: expected a non-empty list of distinct entries")
    for u in users:
        if u not in ("native", "mix") and u not in PERSONAS:
            raise StudyError(f"users: {u!r} is not native|mix|{'|'.join(PERSONAS)}")
    gpus = raw.get("gpus", [0, 1, 3])
    if (not isinstance(gpus, list) or not gpus or len(set(gpus)) != len(gpus)
            or not all(isinstance(g, int) and not isinstance(g, bool) and 0 <= g <= 3 for g in gpus)):
        raise StudyError("gpus: expected a list of distinct GPU indices 0-3")
    ids = raw.get("task_ids")
    if ids is not None and (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids)):
        raise StudyError("task_ids: expected a non-empty list of distinct strings")
    det = raw.get("deterministic", True)
    if not isinstance(det, bool):
        raise StudyError("deterministic: expected bool")
    split = raw.get("split", "train")
    if split not in register.SPLITS:
        raise StudyError(f"split: one of {register.SPLITS}")
    prefix = raw.get("tag_prefix", "st")
    if not isinstance(prefix, str) or not prefix.isalnum():
        raise StudyError("tag_prefix: alphanumeric string")
    host = raw.get("host", "rlab1")
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        raise StudyError("host: expected a hostname")
    for k in ("sec_per_episode", "startup_s"):
        v = raw.get(k, 45 if k == "sec_per_episode" else 180)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise StudyError(f"{k}: expected a positive number")
    return Study(
        name=name, arms=tuple(arms), users=tuple(users), hold_job=_posint(raw, "hold_job", 526576), gpus=tuple(gpus),
        max_concurrent=_posint(raw, "max_concurrent", 2), deterministic=det, split=split, task_ids=tuple(ids) if ids else None,
        tau2_timeout_s=_posint(raw, "tau2_timeout_s", None), eta_min=_posint(raw, "eta_min", None), tag_prefix=prefix,
        mem_gb=_posint(raw, "mem_gb", 40), host=host, sec_per_episode=float(raw.get("sec_per_episode", 45)),
        startup_s=float(raw.get("startup_s", 180)))


def resolve_study_path(p: str | Path) -> Path:
    q = Path(p)
    if q.exists():
        return q.resolve()
    for cand in (STUDIES_DIR / q, STUDIES_DIR / f"{q}.yaml"):
        if cand.exists():
            return cand
    raise StudyError(f"study not found: {p}")


def load_study(path: str | Path) -> Study:
    p = resolve_study_path(path)
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise StudyError(f"{p}: invalid YAML: {e}") from e
    return parse_study(raw, stem=p.stem)


# ----------------------------------------------------------------------------------------------------------------- expansion
@dataclass(frozen=True)
class Step:
    arm: str  # native config name minus `_native`
    user: str  # "native" or a persona
    condition: str  # native | mix | persona
    config_name: str
    tag: str
    n_expected: int
    task_ids: tuple[str, ...] | None  # None = whole split (no override)
    derived_text: str | None = None  # None = the base config file is used as is

    @property
    def config_path(self) -> Path:
        return register.RUNS_DIR / f"{self.config_name}.yaml"


def split_task_ids(split: str) -> list[str]:
    _, split_json = paths.telecom_catalog_paths()
    with open(split_json) as f:
        return sorted(json.load(f)[split])


def round_robin(task_ids: Iterable[str], personas: tuple[str, ...] = PERSONAS) -> dict[str, list[str]]:
    """Deal the SORTED task ids to the personas in order (task i -> personas[i % len]). Deterministic."""
    out: dict[str, list[str]] = {p: [] for p in personas}
    for i, t in enumerate(sorted(task_ids)):
        out[personas[i % len(personas)]].append(t)
    return out


def arm_label(config_name: str) -> str:
    return config_name[: -len("_native")] if config_name.endswith("_native") else config_name


def _derive_text(base: dict, name: str, user: str, ids: tuple[str, ...] | None, split: str, note: str) -> str:
    d = {k: v for k, v in base.items() if k != "num_tasks"} if ids else dict(base)
    d.update(name=name, user=user, split=split, status="ready")
    if ids:
        d["task_ids"] = list(ids)
    header = (f"# [telecom_boed] run config (immutable once used: add a new named config, never edit). GENERATED by eval/runner.py (W4-S16).\n"
              f"# Derived from {base['name']}: {note}\n")
    return header + yaml.safe_dump(d, sort_keys=False, width=10**6, default_flow_style=None)


def _load_base(name: str) -> tuple[dict, register.RunSpec]:
    p = register.RUNS_DIR / f"{name}.yaml"
    if not p.is_file():
        raise StudyError(f"arm {name!r}: no run config {p}")
    spec, text = register.load_run_config(p)
    if spec.user not in register.NATIVE_USERS:
        raise StudyError(f"arm {name!r} must be a native-user config (got user {spec.user!r}); personas are derived")
    return yaml.safe_load(text), spec


def _tag(study: Study, arm_idx: int, code: str) -> str:
    tag = f"{study.tag_prefix}a{arm_idx}{code}"
    if len(tag) > MAX_TAG:
        raise StudyError(f"tag {tag!r} longer than {MAX_TAG} chars (socket path limit); shorten tag_prefix")
    return tag


def expand(study: Study) -> list[Step]:
    """Study -> ordered steps. Pure (writes nothing); derived configs are validated by parsing their text."""
    all_split_ids = split_task_ids(study.split)
    if study.task_ids:
        unknown = sorted(set(study.task_ids) - set(all_split_ids))
        if unknown:
            raise StudyError(f"task_ids not in split {study.split!r}: {unknown[:3]}")
    steps: list[Step] = []
    for ai, arm_cfg in enumerate(study.arms, start=1):
        base, bspec = _load_base(arm_cfg)
        label = arm_label(arm_cfg)
        ids = tuple(sorted(study.task_ids)) if study.task_ids else (tuple(sorted(bspec.task_ids)) if bspec.task_ids else None)
        universe = list(ids) if ids else all_split_ids
        override = study.task_ids is not None or bspec.split != study.split  # native then needs a derived config too
        for user in study.users:
            if user == "native":
                n_ids = len(universe) if (ids or not bspec.num_tasks) else bspec.num_tasks
                if override:
                    name = f"{label}__native"
                    text = _derive_text(base, name, "user_simulator", ids, study.split, f"native user on study {study.name!r} task list")
                    steps.append(Step(label, "native", "native", name, _tag(study, ai, "nat"), len(universe) * bspec.num_trials,
                                      ids, derived_text=text))
                else:
                    steps.append(Step(label, "native", "native", arm_cfg, _tag(study, ai, "nat"), n_ids * bspec.num_trials, ids))
                continue
            if bspec.num_tasks and not ids:
                raise StudyError(f"arm {arm_cfg}: num_tasks set; a persona partition needs explicit ids")
            if user == "mix":
                parts = {p: tuple(v) for p, v in round_robin(universe).items()}
                cond = "mix"
                codes = {p: PERSONA_CODE[p] for p in PERSONAS}
            else:
                parts = {user: tuple(sorted(universe))}
                cond = "persona"
                codes = {user: PERSONA_CODE[user] + "F"}
            for persona, pids in parts.items():
                if not pids:
                    continue
                name = f"{label}__{persona}" if cond == "mix" else f"{label}__{persona}_full"
                note = (f"user boed-user-{persona}, {len(pids)} of {len(universe)} {study.split} tasks (round-robin over sorted ids)"
                        if cond == "mix" else f"user boed-user-{persona}, all {len(pids)} tasks")
                text = _derive_text(base, name, f"boed-user-{persona}", pids, study.split, note)
                steps.append(Step(label, persona, cond, name, _tag(study, ai, codes[persona]), len(pids) * bspec.num_trials, pids,
                                  derived_text=text))
    tags = [s.tag for s in steps]
    if len(set(tags)) != len(tags):
        raise StudyError("duplicate step tags")
    for s in steps:  # validate every derived config exactly as launch.py will read it
        text = s.derived_text if s.derived_text is not None else s.config_path.read_text()
        register.parse_run_config(yaml.safe_load(text), stem=s.config_name)
    return steps


def materialize(steps: list[Step]) -> list[Path]:
    """Write the derived configs as NEW files; an existing file with different content is an error (configs are immutable)."""
    written = []
    for s in steps:
        if s.derived_text is None:
            continue
        p = s.config_path
        if p.exists():
            if p.read_text() != s.derived_text:
                raise StudyError(f"{p} exists with different content (configs are immutable; pick a new arm/persona name)")
            continue
        p.write_text(s.derived_text)
        register.load_run_config(p)  # re-validate from disk
        written.append(p)
    return written


def check_step_names(steps: list[Step]) -> None:
    """Optional: fail if an agent/user name is unregistered (imports tau2 through register_all; needs TAU2_DATA_DIR set safely)."""
    register.register_all()
    for s in steps:
        text = s.derived_text if s.derived_text is not None else s.config_path.read_text()
        register.check_names(register.parse_run_config(yaml.safe_load(text), stem=s.config_name))


# ---------------------------------------------------------------------------------------------------------------- estimates
def step_timeouts(study: Study, step: Step) -> tuple[int, int, int]:
    """(tau2_timeout_s, eta_min, srun step limit in seconds) for a step. Fixed study values win; otherwise 3x the episode estimate (>= 1500 s)."""
    est = step.n_expected * study.sec_per_episode
    tau2_t = study.tau2_timeout_s or max(1500, int(math.ceil(3 * est / 60.0)) * 60)
    eta = study.eta_min or int(math.ceil((study.startup_s + est) * 1.3 / 60.0))
    return tau2_t, eta, tau2_t + HEALTH_TIMEOUT_S + 300


def estimate(study: Study, steps: list[Step]) -> dict:
    eps = sum(s.n_expected for s in steps)
    gpu_s = eps * study.sec_per_episode + len(steps) * study.startup_s
    width = min(study.max_concurrent, len(study.gpus))
    return {"steps": len(steps), "episodes": eps, "gpu_h": gpu_s / 3600.0, "wall_h": gpu_s / 3600.0 / width,
            "sec_per_episode": study.sec_per_episode, "startup_s": study.startup_s, "width": width}


def format_plan(study: Study, steps: list[Step]) -> str:
    est = estimate(study, steps)
    lines = [f"study {study.name}: hold {study.hold_job}, gpus {list(study.gpus)}, max_concurrent {study.max_concurrent}, "
             f"deterministic={study.deterministic}, split {study.split}",
             f"{'#':>3} {'tag':<12} {'arm':<18} {'user':<14} {'cond':<8} {'eps':>4}  run_config"]
    for i, s in enumerate(steps, start=1):
        lines.append(f"{i:>3} {s.tag:<12} {s.arm:<18} {s.user:<14} {s.condition:<8} {s.n_expected:>4}  {s.config_name}"
                     f"{'' if s.derived_text is None else ' (derived)'}")
    lines.append(f"steps {est['steps']}  episodes {est['episodes']}  "
                 f"GPU-h {est['gpu_h']:.1f} (= episodes x {est['sec_per_episode']:g} s + steps x {est['startup_s']:g} s vLLM startup; "
                 f"ASSUMPTION: {est['sec_per_episode']:g} s/episode wall amortised, deterministic mode, config max_concurrency)  "
                 f"wall ~{est['wall_h']:.1f} h at width {est['width']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------------ manifest
def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Manifest:
    def __init__(self, study: str, root: Path | None = None):
        self.dir = (root or RESULTS_STUDIES) / study
        self.path = self.dir / "manifest.jsonl"
        self.study = study

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for ln in self.path.read_text().splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass  # torn tail line of a killed runner
        return out

    def append(self, **rec: Any) -> dict:
        rec = {"t": now(), "study": self.study, **rec}
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
        return rec

    def latest(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.records():
            out[r["tag"]] = r
        return out

    def attempts(self, tag: str) -> int:
        """Attempts that actually ran (an attempt that ended 'no free GPU' is reused by the retry)."""
        last: dict[int, dict] = {}
        for r in self.records():
            if r["tag"] == tag and r.get("attempt") is not None:
                last[r["attempt"]] = r
        return sum(1 for r in last.values() if not r.get("busy"))

    def run_dirs(self, tag: str) -> list[str]:
        seen: list[str] = []
        for r in self.records():
            if r["tag"] == tag and r.get("run_dir") and r["run_dir"] not in seen:
                seen.append(r["run_dir"])
        return seen


def run_complete(run_dir: str | Path, n_expected: int) -> bool:
    p = Path(run_dir) / "results.json"
    try:
        sims = json.loads(p.read_text()).get("simulations")
    except (OSError, ValueError):
        return False
    return isinstance(sims, list) and len(sims) == n_expected


# ---------------------------------------------------------------------------------------------------------------------- run
def run_dir_for(study: Study, tag: str, attempt: int) -> Path:
    return RESULTS_RUNS / f"{tag}_{study.hold_job}.a{attempt}"


def gpu_lock_argv(study: Study, step: Step, run_dir: Path) -> list[str]:
    """Local argv: `cdx-agent --gpu-lock` takes one free GPU lock in the hold and runs the launcher (it does the ssh + srun)."""
    tau2_t, eta, limit_s = step_timeouts(study, step)
    env = []
    if study.deterministic:
        env.append("DETERMINISTIC=1")
    env += ["MODE=launch", f"RUN_CONFIG={step.config_path}", f"RUN_TAG={step.tag}", f"TAU2_TIMEOUT_S={tau2_t}", f"RESULT_DIR={run_dir}"]
    return [CDX_AGENT, "--gpu-lock", str(study.hold_job), "--n", "1", "--eta", str(eta), "--host", study.host, "--mem", f"{study.mem_gb}G",
            "--time", str(int(math.ceil(limit_s / 60.0))), "--", "env", *env, "bash", str(LAUNCHER_ABS)]


def read_gpu(run_dir: str | Path) -> int | None:
    """GPU the step ran on, from <run_dir>/job_info.txt ('gpu_index=N', written by the launcher)."""
    try:
        for ln in (Path(run_dir) / "job_info.txt").read_text().splitlines():
            if ln.startswith("gpu_index="):
                return int(ln.split()[0].split("=", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def _default_popen(argv: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "ab")
    return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


@dataclass
class _Live:
    step: Step
    attempt: int
    run_dir: Path
    proc: Any
    t0: float = field(default_factory=time.monotonic)


def run_study(study: Study, steps: list[Step], *, dry_run: bool = False, popen: Callable[[list[str], Path], Any] = _default_popen,
              sleep: Callable[[float], None] = time.sleep, poll_s: float = 20.0, busy_cooldown_s: float = 120.0,
              hold_check: Callable[[Study], str] | None = None, manifest: Manifest | None = None,
              out: Callable[[str], None] = print) -> dict:
    """Run the queue. dry_run prints the commands that would launch and touches nothing (no files, no manifest, no ssh)."""
    mf = manifest or Manifest(study.name)
    latest = mf.latest()
    pending: list[Step] = []
    for s in steps:
        dirs = mf.run_dirs(s.tag)
        done_dir = next((d for d in dirs if run_complete(d, s.n_expected)), None)
        if done_dir:
            if latest.get(s.tag, {}).get("state") != "done" and not dry_run:
                mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=s.tag, gpu=latest.get(s.tag, {}).get("gpu"), state="done",
                          run_dir=done_dir, exit_rc=latest.get(s.tag, {}).get("exit_rc"), attempt=latest.get(s.tag, {}).get("attempt"),
                          n_expected=s.n_expected, condition=s.condition, note="found complete on resume")
            continue
        pending.append(s)
    skipped = len(steps) - len(pending)
    out(f"{len(steps)} steps: {skipped} already complete, {len(pending)} to run")
    if dry_run:
        for s in pending:
            att = mf.attempts(s.tag) + 1
            out(f"[dry-run] {s.tag}: {' '.join(shlex.quote(a) for a in gpu_lock_argv(study, s, run_dir_for(study, s.tag, att)))}")
        return {"skipped": skipped, "launched": 0, "dry_run": True}
    if not pending:
        return {"skipped": skipped, "launched": 0}
    mf.dir.mkdir(parents=True, exist_ok=True)
    lock = open(mf.dir / "runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        raise StudyError(f"another runner holds {mf.dir / 'runner.lock'}") from e
    if hold_check is not None:
        state = hold_check(study)
        if state != "RUNNING":
            raise StudyError(f"hold job {study.hold_job} is not RUNNING (squeue says {state!r})")
    materialize(pending)
    for s in pending:
        mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=s.tag, gpu=None, state="queued", run_dir=None, exit_rc=None,
                  attempt=None, n_expected=s.n_expected, condition=s.condition)
    queue = list(pending)
    live: dict[str, _Live] = {}  # tag -> live step (the GPU is chosen by the wrapper and read back from job_info.txt)
    cooldown_until = 0.0  # monotonic time before which nothing new launches (after a "no free GPU" exit)
    busy_tries: dict[str, int] = {}
    launched = 0
    summary = {"skipped": skipped, "launched": 0, "done": 0, "failed": 0}
    try:
        while queue or live:
            for tag, lv in list(live.items()):
                rc = lv.proc.poll()
                if rc is None:
                    continue
                del live[tag]
                s = lv.step
                if rc in GPU_BUSY_RCS and busy_tries.get(tag, 0) < MAX_BUSY_RETRIES:  # wrapper found no free GPU: queued, not failed
                    busy_tries[tag] = busy_tries.get(tag, 0) + 1
                    cooldown_until = time.monotonic() + busy_cooldown_s
                    queue.insert(0, s)
                    mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=tag, gpu=None, state="queued", run_dir=None,
                              exit_rc=rc, attempt=lv.attempt, n_expected=s.n_expected, condition=s.condition, busy=True,
                              note=f"no free GPU (rc {rc}); retry in {busy_cooldown_s:g} s")
                    out(f"{tag}: no free GPU (rc {rc}); queued, retry later")
                    continue
                ok = rc == 0 and run_complete(lv.run_dir, s.n_expected)
                g = read_gpu(lv.run_dir)
                mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=tag, gpu=g, state="done" if ok else "failed",
                          run_dir=str(lv.run_dir), exit_rc=rc, attempt=lv.attempt, n_expected=s.n_expected, condition=s.condition,
                          wall_s=round(time.monotonic() - lv.t0))
                summary["done" if ok else "failed"] += 1
                out(f"{tag}: {'done' if ok else 'FAILED'} rc={rc} gpu={g}")
            while queue and len(live) < study.max_concurrent and time.monotonic() >= cooldown_until:
                s = queue.pop(0)
                att = mf.attempts(s.tag) + 1
                rd = run_dir_for(study, s.tag, att)
                log = mf.dir / "logs" / f"{s.tag}.a{att}.log"
                proc = popen(gpu_lock_argv(study, s, rd), log)
                live[s.tag] = _Live(s, att, rd, proc)
                launched += 1
                mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=s.tag, gpu=None, state="running", run_dir=str(rd),
                          exit_rc=None, attempt=att, n_expected=s.n_expected, condition=s.condition)
                out(f"{s.tag}: launched (attempt {att})")
            if queue or live:
                sleep(poll_s)
    except KeyboardInterrupt:
        for lv in live.values():
            try:
                lv.proc.terminate()
            except Exception:
                pass
            s = lv.step
            mf.append(arm=s.arm, user=s.user, run_config=s.config_name, tag=s.tag, gpu=None, state="failed", run_dir=str(lv.run_dir),
                      exit_rc=-15, attempt=lv.attempt, n_expected=s.n_expected, condition=s.condition,
                      note="runner interrupted; the remote srun step may still run (cdx-agent --gpu-lock --status / squeue)")
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    summary["launched"] = launched
    return summary


def status(study_name: str, manifest: Manifest | None = None) -> str:
    mf = manifest or Manifest(study_name)
    latest = mf.latest()
    if not latest:
        return f"study {study_name}: no manifest at {mf.path}"
    counts: dict[str, int] = {}
    for r in latest.values():
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    lines = [f"study {study_name}: {len(latest)} steps  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))]
    for tag, r in latest.items():
        lines.append(f"{tag:<12} {r['state']:<8} {str(r.get('arm')):<18} {str(r.get('user')):<14} gpu={r.get('gpu')} rc={r.get('exit_rc')} "
                     f"n={r.get('n_expected')} {Path(r['run_dir']).name if r.get('run_dir') else ''}")
    return "\n".join(lines)
