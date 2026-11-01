"""[telecom_boed] W3-S10: run-config schema, launcher guards, registration idempotence, --tau2-cli passthrough."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from telecom_boed import paths
from telecom_boed.tau2_ext import launch, register

RUNS = sorted((paths.CONFIG_DIR / "runs").glob("*.yaml"))
P3_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
GOOD = {"name": "t_good", "llm": {"model": "test/none"}}


def _bad(**kw) -> dict:
    d = copy.deepcopy(GOOD)
    d.update(kw)
    return d


# ------------------------------------------------------------------------------------------------ schema
def test_all_shipped_run_configs_validate():
    assert len(RUNS) >= 15
    for p in RUNS:
        spec, text = register.load_run_config(p)
        assert spec.name == p.stem and register.sha256_text(text)
        if spec.status == "template":
            assert p.stem.startswith("tmpl_")  # templates are schema-checked only
        assert spec.eval_final == (spec.split == "test")


def test_template_matrix_complete():
    names = {p.stem for p in RUNS if p.stem.startswith("tmpl_")}
    objs = ("reach", "dug", "kstep", "eig", "heuristic")
    want = {f"tmpl_boed_controller_{o}_{u}" for o in objs for u in ("native", "persona")}
    want |= {"tmpl_boed_plain_native", "tmpl_boed_plain_persona"}
    assert names == want
    for n in want:
        spec, _ = register.load_run_config(paths.CONFIG_DIR / "runs" / f"{n}.yaml")
        assert spec.status == "template" and spec.split == "train"
        assert (spec.user == "user_simulator") == n.endswith("native")
        assert spec.agent == ("boed-plain" if "plain" in n else "boed-controller")


@pytest.mark.parametrize("mutate, msg", [
    (dict(bogus=1), "unknown keys"),
    (dict(workers=2), "workers"),
    (dict(domain="airline"), "domain"),
    (dict(split="dev"), "split"),
    (dict(split="test"), "eval_final"),
    (dict(split="base"), "eval_final"),
    (dict(agent="llm_agent_gt"), "refused"),
    (dict(agent="llm_agent", agent_kwargs={"x": 1}), "native agent"),
    (dict(task_ids=["a"], num_tasks=2), "mutually exclusive"),
    (dict(task_ids=[]), "task_ids"),
    (dict(num_trials=0), "num_trials"),
    (dict(max_steps=True), "max_steps"),
    (dict(boed_config="configs/nope.yaml"), "boed_config"),
    (dict(llm={"model": "m", "api_base": "x"}), "unknown keys"),
    (dict(llm={"temperature": 0.0}), "model"),
    (dict(status="done"), "status"),
    (dict(name="bad name"), "name"),
])
def test_bad_configs_rejected(mutate, msg):
    with pytest.raises(register.ConfigError, match=msg):
        register.parse_run_config(_bad(**mutate))


def test_missing_llm_and_stem_mismatch_and_not_mapping():
    with pytest.raises(register.ConfigError, match="llm: required"):
        register.parse_run_config({"name": "x"})
    with pytest.raises(register.ConfigError, match="file stem"):
        register.parse_run_config(GOOD, stem="other")
    with pytest.raises(register.ConfigError, match="mapping"):
        register.parse_run_config([1, 2])


def test_good_config_defaults_and_test_split_with_eval_final():
    s = register.parse_run_config(GOOD)
    assert (s.agent, s.user, s.domain, s.split, s.num_trials, s.seed, s.max_steps, s.max_errors) == (
        "llm_agent", "user_simulator", "telecom", "train", 1, 300, 200, 10)
    assert register.parse_run_config(_bad(split="test", eval_final=True)).eval_final


# ------------------------------------------------------------------------------------------------ llm args / side channel
def test_llm_args_native_vs_boed_side_channel():
    env = {"VLLM_API_BASE": "http://h:1/v1"}
    nat = register.parse_run_config(_bad(llm={"model": "hosted_vllm/m", "extra_body": P3_EXTRA_BODY}))
    a = register.llm_args_for(nat, "agent", env)
    assert a == {"temperature": 0.0, "api_base": "http://h:1/v1", "api_key": "local", "extra_body": P3_EXTRA_BODY}
    boed = replace(nat, agent="boed-controller", agent_kwargs={"objective": "reach"}, user="boed-user-terse")
    ab, ub = register.llm_args_for(boed, "agent", env), register.llm_args_for(boed, "user", env)
    la, payload = register.split_boed_args(ab)
    assert la == a and "_boed" in ab  # not mutated
    assert payload["agent_kwargs"] == {"objective": "reach"} and payload["run"] == "t_good"
    assert payload["boed_config_sha256"] and register.split_boed_args(ub)[1]["user"] == "boed-user-terse"
    assert register.split_boed_args(a) == (a, {})
    with pytest.raises(register.ConfigError, match="VLLM_API_BASE"):
        register.llm_args_for(nat, "agent", {})


def test_user_llm_override():
    s = register.parse_run_config(_bad(llm={"model": "test/a"}, user_llm={"model": "test/b", "temperature": 0.7}))
    assert register.llm_args_for(s, "user", {})["temperature"] == 0.7
    assert register.llm_args_for(s, "agent", {})["temperature"] == 0.0


# ------------------------------------------------------------------------------------------------ TextRunConfig mapping
def test_test40_config_reproduces_p3_cli_baseline(monkeypatch):
    """launch's TextRunConfig for plain_native_test40 == the TextRunConfig tau2's own CLI builds for the P3 args."""
    from tau2 import cli
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.work import parse_provider_limits

    base = "http://127.0.0.1:8000/v1"
    monkeypatch.setenv("VLLM_API_BASE", base)
    spec, _ = register.load_run_config(paths.CONFIG_DIR / "runs" / "plain_native_test40.yaml")
    mine = launch.build_text_run_config(spec, "SAVE")
    llm_args = json.dumps({"api_base": base, "api_key": "local", "temperature": 0.0,
                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    ap = argparse.ArgumentParser()
    cli.add_run_args(ap)
    a = ap.parse_args(["--domain", "telecom", "--task-split-name", "test", "--num-trials", "4", "--max-concurrency", "4",
                       "--max-steps", "200", "--seed", "300", "--agent-llm", "hosted_vllm/Qwen/Qwen3-8B",
                       "--agent-llm-args", llm_args, "--user-llm", "hosted_vllm/Qwen/Qwen3-8B", "--user-llm-args", llm_args,
                       "--save-to", "SAVE"])
    theirs = TextRunConfig(
        domain=a.domain, task_set_name=a.task_set_name, task_split_name=a.task_split_name, task_ids=a.task_ids,
        num_tasks=a.num_tasks, llm_user=a.user_llm, llm_args_user=a.user_llm_args, num_trials=a.num_trials,
        max_errors=a.max_errors, timeout=a.timeout, save_to=a.save_to, max_concurrency=a.max_concurrency,
        workers=a.workers, provider_limits=parse_provider_limits(a.provider_limit), seed=a.seed, log_level=a.log_level,
        verbose_logs=a.verbose_logs, max_retries=a.max_retries, retry_delay=a.retry_delay, auto_resume=a.auto_resume,
        auto_review=a.auto_review, review_mode=a.review_mode, review_model=a.review_model,
        hallucination_retries=a.hallucination_retries, retrieval_config=a.retrieval_config,
        retrieval_config_kwargs=a.retrieval_config_kwargs, agent=a.agent, llm_agent=a.agent_llm,
        llm_args_agent=a.agent_llm_args, user=a.user, max_steps=a.max_steps,
        enforce_communication_protocol=a.enforce_communication_protocol)
    assert mine.model_dump() == theirs.model_dump()
    assert spec.eval_final and spec.num_trials == 4 and spec.seed == 300 and spec.max_concurrency == 4


def test_workflow_config_uses_workflow_domain():
    spec, _ = register.load_run_config(paths.CONFIG_DIR / "runs" / "workflow_native_train.yaml")
    assert spec.domain == "telecom-workflow" and spec.split == "train" and not spec.eval_final


# ------------------------------------------------------------------------------------------------ guards
def test_test_split_guard_without_eval_final():
    spec, _ = register.load_run_config(paths.CONFIG_DIR / "runs" / "plain_native_test40.yaml")
    launch._guard_config(spec)  # eval_final: true -> ok
    with pytest.raises(SystemExit, match="eval-final|eval_final"):
        launch._guard_config(replace(spec, eval_final=False))
    with pytest.raises(SystemExit, match="eval-final"):
        launch._guard(["--task-split-name", "test"], False)


def test_run_config_refuses_bad_config_file(tmp_path):
    p = tmp_path / "t_bad.yaml"
    p.write_text(yaml.safe_dump({"name": "t_bad", "llm": {"model": "m"}, "split": "test"}))
    with pytest.raises(SystemExit, match="eval_final"):
        launch.run_config(p)


def test_workers_guard():
    with pytest.raises(SystemExit, match="workers"):
        launch._guard(["--workers", "2"], False)
    launch._guard(["--workers", "0"], False)
    with pytest.raises(register.ConfigError, match="workers"):
        register.parse_run_config(_bad(workers=0))


def test_data_dir_guards(monkeypatch, tmp_path):
    monkeypatch.delenv("TAU2_DATA_DIR")
    with pytest.raises(SystemExit, match="unset"):
        launch._guard([], False)
    monkeypatch.setenv("TAU2_DATA_DIR", str(paths.tau2_root() / "data"))
    with pytest.raises(SystemExit, match="inside the pinned clone"):
        launch._guard([], False)
    monkeypatch.setenv("TAU2_DATA_DIR", str(tmp_path))  # differs from the data dir tau2 was imported with
    with pytest.raises(SystemExit, match="imported before"):
        launch._guard([], False)


def test_existing_names_check_after_register():
    register.register_all()
    spec, _ = register.load_run_config(paths.CONFIG_DIR / "runs" / "plain_native_train.yaml")
    register.check_names(spec)
    with pytest.raises(register.ConfigError, match="not registered"):
        register.check_names(replace(spec, agent="no-such-agent"))
    with pytest.raises(register.ConfigError, match="not registered"):
        register.check_names(replace(spec, user="no-such-user"))


# ------------------------------------------------------------------------------------------------ registration
def test_register_all_idempotent():
    a = register.register_all()
    assert register.register_all() == a
    # a cold second pass (cache cleared) must also be a no-op for names we already put in tau2's registry
    saved = register._REGISTERED
    register._REGISTERED = None
    try:
        assert register.register_all() == a
    finally:
        register._REGISTERED = saved


def test_idempotent_proxy_semantics():
    from tau2.registry import registry
    from tau2.user.user_simulator_base import HalfDuplexUser

    class U1(HalfDuplexUser):
        def get_init_state(self, message_history=None): ...
        def generate_next_message(self, message, state): ...

    class U2(U1):
        pass

    def f1(tools, domain_policy, **kw): ...

    def f2(tools, domain_policy, **kw): ...

    px = register._IdempotentRegistry(registry)
    px.register_user(U1, "w3s10-u")
    px.register_user(U1, "w3s10-u")  # same object -> no-op (tau2 itself would raise ValueError)
    with pytest.raises(ValueError):
        px.register_user(U2, "w3s10-u")  # different object under the same name still raises
    px.register_agent_factory(f1, "w3s10-a")
    px.register_agent_factory(f1, "w3s10-a")
    with pytest.raises(ValueError):
        px.register_agent_factory(f2, "w3s10-a")
    assert px.users == ["w3s10-u"] and px.agents == ["w3s10-a"]


def test_missing_plugin_skipped_other_import_errors_raise(monkeypatch, tmp_path, capsys):
    (tmp_path / "w3s10_ok.py").write_text("def register(registry):\n    return ['w3s10-ok']\n")
    (tmp_path / "w3s10_broken.py").write_text("import w3s10_not_a_real_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(register, "_REGISTERED", None)
    monkeypatch.setattr(register, "PLUGIN_MODULES", ("w3s10_missing_mod", "w3s10_ok"))
    assert register.register_all() == ["w3s10-ok"]
    assert "w3s10_missing_mod not present yet, skipped" in capsys.readouterr().err
    monkeypatch.setattr(register, "_REGISTERED", None)
    monkeypatch.setattr(register, "PLUGIN_MODULES", ("w3s10_broken",))
    with pytest.raises(ModuleNotFoundError, match="w3s10_not_a_real_dependency"):
        register.register_all()


# ------------------------------------------------------------------------------------------------ CLI subprocess
def _sub(args: list[str], tmp_path: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "TELECOM_RUN_DIR": str(tmp_path / "rd"), **env}
    return subprocess.run([sys.executable, "-m", "telecom_boed.tau2_ext.launch", *args], capture_output=True, text=True,
                          cwd=tmp_path, env=e, timeout=300)


def test_tau2_cli_passthrough_still_works(tmp_path):
    r = _sub(["--tau2-cli", "--", "run", "--help"], tmp_path)
    assert r.returncode == 0, r.stderr[-500:]
    assert "--task-split-name" in r.stdout
    meta = json.loads((tmp_path / "rd" / "run_meta.json").read_text())
    assert meta["tau2_args"] == ["run", "--help"] and meta["patches"] and "run_config" not in meta


def test_tau2_cli_guards_via_subprocess(tmp_path):
    r = _sub(["--tau2-cli", "--", "run", "--task-split-name", "test"], tmp_path)
    assert r.returncode != 0 and "--eval-final" in r.stderr
    r = _sub(["--tau2-cli", "--", "run", "--workers", "3"], tmp_path)
    assert r.returncode != 0 and "workers" in r.stderr
    r = _sub(["--tau2-cli", "--", "run", "--help"], tmp_path, TAU2_DATA_DIR="")
    assert r.returncode != 0 and "TAU2_DATA_DIR" in r.stderr


def test_config_mode_cli_error_paths(tmp_path):
    bad = tmp_path / "t_cli_bad.yaml"
    bad.write_text(textwrap.dedent("""\
        name: t_cli_bad
        llm: {model: m}
        split: test
        """))
    r = _sub(["--config", str(bad)], tmp_path)
    assert r.returncode != 0 and "eval_final" in r.stderr
    nope = tmp_path / "t_cli_nope.yaml"
    nope.write_text("name: t_cli_nope\nagent: no-such-agent\nllm: {model: m}\n")
    r = _sub(["--config", str(nope)], tmp_path)
    assert r.returncode != 0 and "not registered" in r.stderr  # unregistered names are refused before any run
