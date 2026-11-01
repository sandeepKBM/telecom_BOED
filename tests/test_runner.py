"""[telecom_boed] W4-S16: study runner (expansion, persona split, derived configs, dry-run, manifest resume). No GPU, no ssh."""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest
import yaml

from telecom_boed import paths
from telecom_boed.eval import runner
from telecom_boed.tau2_ext import register

P4 = paths.CONFIG_DIR / "studies" / "p4_train.yaml"
MINI = Path(__file__).parent / "fixtures" / "W4-S16" / "mini_study.yaml"


@pytest.fixture(scope="module")
def p4():
    study = runner.load_study(P4)
    return study, runner.expand(study)


def test_p4_plan_counts(p4):
    study, steps = p4
    assert (study.hold_job, study.gpus, study.max_concurrent, study.deterministic) == (526576, (0, 1, 3), 2, True)
    assert len(study.arms) == 8
    assert len(steps) == 8 * 7  # native + 6 personas per arm
    assert sum(s.n_expected for s in steps) == 8 * 148 == 1184
    tags = [s.tag for s in steps]
    assert len(set(tags)) == len(tags) and max(map(len, tags)) <= runner.MAX_TAG
    est = runner.estimate(study, steps)
    assert est["steps"] == 56 and est["episodes"] == 1184
    assert est["gpu_h"] == pytest.approx((1184 * 45 + 56 * 180) / 3600)
    text = runner.format_plan(study, steps)
    assert "episodes 1184" in text and "ASSUMPTION: 45 s/episode" in text


def test_arms_cover_the_p4_design(p4):
    study, _ = p4
    specs = {a: register.load_run_config(register.RUNS_DIR / f"{a}.yaml")[0] for a in study.arms}
    assert all((s.split, s.num_trials, s.seed, s.max_concurrency, s.max_steps, s.max_errors) == ("train", 1, 300, 4, 200, 10)
               for s in specs.values())
    assert specs["p4_plain_native"].agent == "llm_agent" and specs["p4_plain_native"].domain == "telecom"
    assert specs["p4_workflow_native"].agent == "llm_agent" and specs["p4_workflow_native"].domain == "telecom-workflow"
    assert specs["p4_plain_repair_native"].agent == "boed-plain" and specs["p4_plain_repair_native"].agent_kwargs["repair_invalid_user_tools"]
    scorers = {s.agent_kwargs["scorer"] for a, s in specs.items() if s.agent == "boed-controller"}
    assert scorers == {"reach", "dug", "kstep", "eig", "heuristic"}
    assert all(s.agent_kwargs["repair_invalid_user_tools"] for s in specs.values() if s.agent == "boed-controller")


def test_round_robin_split_deterministic(train_ids):
    a = runner.round_robin(train_ids)
    shuffled = list(train_ids)
    random.Random(1).shuffle(shuffled)
    assert runner.round_robin(shuffled) == a
    assert list(a) == list(runner.PERSONAS)
    assert [len(a[p]) for p in runner.PERSONAS] == [13, 13, 12, 12, 12, 12]
    flat = [t for p in runner.PERSONAS for t in a[p]]
    assert sorted(flat) == sorted(train_ids) and len(set(flat)) == 74
    s = sorted(train_ids)
    assert a["cooperative"][:2] == [s[0], s[6]] and a["terse"][0] == s[1] and a["delayed"][0] == s[5]


def test_persona_table_matches_personas_module():
    from telecom_boed.tau2_ext import personas
    assert runner.PERSONAS == personas.PERSONAS and set(runner.PERSONA_CODE) == set(personas.PERSONAS)


def test_pairs_share_task_ids_and_personas(p4, train_ids):
    _, steps = p4
    by = {}
    for s in steps:
        by.setdefault(s.user, {})[s.arm] = s.task_ids
    for persona in runner.PERSONAS:
        assert len({v for v in by[persona].values()}) == 1  # identical ids for every arm
    assert all(v is None for v in by["native"].values())  # native = whole split
    assert sorted(t for p in runner.PERSONAS for t in by[p]["p4_plain"]) == sorted(train_ids)


def test_generated_configs_are_on_disk_and_validate(p4):
    _, steps = p4
    derived = [s for s in steps if s.derived_text is not None]
    assert len(derived) == 48
    for s in derived:
        assert s.config_path.read_text() == s.derived_text  # `gen-configs` has been run; files are immutable
        spec, _ = register.load_run_config(s.config_path)
        assert spec.name == s.config_name and spec.user == f"boed-user-{s.user}" and spec.task_ids == s.task_ids
        assert spec.num_tasks is None and spec.split == "train"
    for s in steps:
        if s.derived_text is None:
            assert register.load_run_config(s.config_path)[0].task_ids is None


def test_generated_configs_check_names(p4):
    _, steps = p4
    runner.check_step_names(steps)  # all agents/users registered (imports tau2 via register_all)


def test_materialize_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(register, "RUNS_DIR", tmp_path)
    for n in ("p4_plain_native", "p4_ctrl_reach_native"):
        shutil.copy(paths.CONFIG_DIR / "runs" / f"{n}.yaml", tmp_path / f"{n}.yaml")
    steps = runner.expand(runner.load_study(MINI))
    assert len(runner.materialize(steps)) == len([s for s in steps if s.derived_text])
    assert runner.materialize(steps) == []  # idempotent
    s = next(x for x in steps if x.derived_text)
    s.config_path.write_text(s.derived_text + "# edited\n")
    with pytest.raises(runner.StudyError, match="immutable"):
        runner.materialize(steps)


def test_study_validation():
    ok = yaml.safe_load(MINI.read_text())
    for bad in ({"bogus": 1}, {"users": ["nobody"]}, {"gpus": [0, 7]}, {"arms": []}, {"max_concurrent": 0}, {"task_ids": []}):
        with pytest.raises(runner.StudyError):
            runner.parse_study({**ok, **bad})
    long_tag = runner.parse_study({**ok, "tag_prefix": "waytoolongprefix"})
    with pytest.raises(runner.StudyError, match="longer than"):
        runner.expand(long_tag)


def test_gpu_lock_command(p4):
    study, steps = p4
    argv = runner.gpu_lock_argv(study, steps[0], Path("/x/runs/p4a1nat_526576.a1"))
    assert argv[:9] == ["cdx-agent", "--gpu-lock", "526576", "--n", "1", "--eta", argv[6], "--host", "rlab1"]
    assert argv[argv.index("--mem") + 1] == "40G" and "--time" in argv and "--" in argv
    cmd = argv[argv.index("--") + 1:]
    assert cmd[:4] == ["env", "DETERMINISTIC=1", "MODE=launch", f"RUN_CONFIG={steps[0].config_path}"]
    assert "RUN_TAG=p4a1nat" in cmd and "RESULT_DIR=/x/runs/p4a1nat_526576.a1" in cmd
    assert not any(c.startswith("GPU_INDEX=") for c in cmd)  # the wrapper picks the GPU
    assert cmd[-2:] == ["bash", "/common/users/ss5772/ddl/telecom_boed/slurm/local_tau2_vllm.sh"]  # absolute: step starts in $HOME
    t, eta, limit_s = runner.step_timeouts(study, steps[0])
    assert int(argv[argv.index("--time") + 1]) * 60 >= limit_s > t


# ------------------------------------------------------------------------------------------------------ execution (fake ssh)
class FakeProc:
    """Stands in for `cdx-agent --gpu-lock ...`: takes the lowest free GPU of 0-3 like the wrapper, or exits 3 when told to."""

    def __init__(self, fake, argv, ticks, rc):
        self.fake, self.ticks, self.rc = fake, ticks, rc
        self.polls = 0
        cmd = argv[argv.index("--") + 1:]
        self.env = dict(kv.split("=", 1) for kv in cmd if "=" in kv)
        self.gpu = None
        if rc not in runner.GPU_BUSY_RCS:
            self.gpu = min(g for g in range(4) if g not in fake.gpus_live)
            fake.gpus_live.append(self.gpu)
            fake.live += 1
            fake.max_live = max(fake.max_live, fake.live)

    def poll(self):
        self.polls += 1
        if self.polls < self.ticks:
            return None
        if self.gpu is not None:
            self.fake.live -= 1
            self.fake.gpus_live.remove(self.gpu)
            d = Path(self.env["RESULT_DIR"])
            d.mkdir(parents=True, exist_ok=True)
            (d / "job_info.txt").write_text(f"gpu_index={self.gpu} CUDA_VISIBLE_DEVICES={self.gpu}\n")
            if self.rc == 0:
                n = self.fake.n_for[self.env["RUN_TAG"]]
                (d / "results.json").write_text(json.dumps({"simulations": [{"id": i} for i in range(n)]}))
        return self.rc

    def terminate(self):
        pass


class FakeLauncher:
    def __init__(self, steps, rcs=None):
        self.n_for = {s.tag: s.n_expected for s in steps}
        self.rcs = rcs or {}  # tag -> list of rcs per attempt
        self.calls = []
        self.live = self.max_live = 0
        self.gpus_live = []

    def __call__(self, argv, log):
        tag = next(a.split("=")[1] for a in argv if a.startswith("RUN_TAG="))
        self.calls.append((tag, argv))
        seq = self.rcs.get(tag, [0])
        rc = seq[min(sum(1 for t, _ in self.calls if t == tag) - 1, len(seq) - 1)]
        return FakeProc(self, argv, ticks=2, rc=rc)


@pytest.fixture
def mini(tmp_path, monkeypatch):
    runs = tmp_path / "cfg"
    runs.mkdir()
    for n in ("p4_plain_native", "p4_ctrl_reach_native"):
        shutil.copy(paths.CONFIG_DIR / "runs" / f"{n}.yaml", runs / f"{n}.yaml")
    monkeypatch.setattr(register, "RUNS_DIR", runs)
    monkeypatch.setattr(runner, "RESULTS_RUNS", tmp_path / "runs")
    raw = yaml.safe_load(MINI.read_text())
    sp = [t for t in runner.split_task_ids("train")][:20:3]  # 7 tasks -> personas get 2,1,1,1,1,1
    study = runner.parse_study({**raw, "task_ids": sp})
    steps = runner.expand(study)
    mf = runner.Manifest(study.name, root=tmp_path / "studies")
    return study, steps, mf, tmp_path


def _run(study, steps, mf, fake, **kw):
    msgs = []
    res = runner.run_study(study, steps, popen=fake, sleep=lambda s: None, hold_check=lambda st: "RUNNING", manifest=mf,
                           busy_cooldown_s=0, out=msgs.append, **kw)
    return res, msgs


def test_mini_expansion(mini):
    study, steps, _, _ = mini
    assert len(steps) == 2 * 7 and sum(s.n_expected for s in steps) == 2 * 14
    assert all(s.derived_text for s in steps)  # task_ids override: native is derived too


def test_dry_run_executes_nothing(mini):
    study, steps, mf, tmp = mini

    def boom(*a, **k):
        raise AssertionError("dry-run must not launch")

    msgs = []
    res = runner.run_study(study, steps, dry_run=True, popen=boom, sleep=boom, hold_check=boom, manifest=mf, out=msgs.append)
    assert res["launched"] == 0 and res["dry_run"]
    assert sum(m.startswith("[dry-run]") for m in msgs) == len(steps)
    assert not mf.path.exists() and not (tmp / "runs").exists() and not list(register.RUNS_DIR.glob("*__*.yaml"))


def test_run_respects_concurrency_gpus_and_records_manifest(mini):
    study, steps, mf, _ = mini
    fake = FakeLauncher(steps)
    res, _ = _run(study, steps, mf, fake)
    assert res["launched"] == 14 and res["done"] == 14 and res["failed"] == 0
    assert fake.max_live == 2
    assert all(a[0] == "cdx-agent" and "DETERMINISTIC=1" in a and "MODE=launch" in a for _, a in fake.calls)
    latest = mf.latest()
    assert len(latest) == 14 and {r["state"] for r in latest.values()} == {"done"}
    states = [r["state"] for r in mf.records()]
    assert states.count("queued") == 14 and states.count("running") == 14 and states.count("done") == 14
    r = latest[steps[0].tag]
    assert {"t", "study", "arm", "user", "run_config", "tag", "gpu", "state", "run_dir", "exit_rc"} <= set(r)
    assert r["exit_rc"] == 0 and runner.run_complete(r["run_dir"], r["n_expected"])
    assert r["gpu"] in (0, 1) and runner.read_gpu(r["run_dir"]) == r["gpu"]  # recorded from the run dir's job_info.txt
    assert "done=14" in runner.status(study.name, manifest=mf)


def test_manifest_resumable(mini):
    study, steps, mf, _ = mini
    _run(study, steps, mf, FakeLauncher(steps))
    # all complete -> nothing launches
    fake2 = FakeLauncher(steps)
    res, _ = _run(study, steps, mf, fake2)
    assert res["launched"] == 0 and res["skipped"] == 14 and not fake2.calls
    # one run loses its results.json (incomplete sims) -> only that step reruns, as attempt 2
    victim = steps[3]
    rd = Path(mf.latest()[victim.tag]["run_dir"])
    (rd / "results.json").write_text(json.dumps({"simulations": []}))
    fake3 = FakeLauncher(steps)
    res, _ = _run(study, steps, mf, fake3)
    assert [t for t, _ in fake3.calls] == [victim.tag] and res["done"] == 1
    last = mf.latest()[victim.tag]
    assert last["attempt"] == 2 and last["run_dir"].endswith(".a2") and last["state"] == "done"


def test_failed_then_resumed_and_gpu_busy_retry(mini):
    study, steps, mf, _ = mini
    bad, busy = steps[0].tag, steps[1].tag
    fake = FakeLauncher(steps, rcs={bad: [1], busy: [3, 0]})
    res, _ = _run(study, steps, mf, fake)
    assert res["failed"] == 1 and res["done"] == 13
    assert mf.latest()[bad]["state"] == "failed" and mf.latest()[bad]["exit_rc"] == 1
    assert sum(1 for t, _ in fake.calls if t == busy) == 2 and mf.latest()[busy]["state"] == "done"
    assert mf.latest()[busy]["attempt"] == 1 and any(r["state"] == "queued" and r.get("busy") for r in mf.records() if r["tag"] == busy)
    assert mf.attempts(bad) == 1
    fake2 = FakeLauncher(steps)
    res, _ = _run(study, steps, mf, fake2)
    assert [t for t, _ in fake2.calls] == [bad] and mf.latest()[bad]["state"] == "done"


def test_hold_must_be_running(mini):
    study, steps, mf, _ = mini
    with pytest.raises(runner.StudyError, match="not RUNNING"):
        runner.run_study(study, steps, popen=FakeLauncher(steps), sleep=lambda s: None, hold_check=lambda st: "", manifest=mf)
