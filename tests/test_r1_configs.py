"""[telecom_boed] P2 (R1) configs: parse, registered names, stated invariants, oracle guard, study expansion. No GPU, no ssh."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from telecom_boed import paths
from telecom_boed.eval import runner
from telecom_boed.tau2_ext import register

ARMS = {"plain": ("llm_agent", "telecom"), "plainrepair": ("boed-plain", "telecom"),
        "workflow": ("llm_agent", "telecom-workflow"), "teacher": ("oracle-teacher", "telecom")}
STAGES = {"pilot": ("r1_train", "", 1), "powered": ("r1_train_powered", "_t4", 4)}


def _specs():
    for arm, (agent, domain) in ARMS.items():
        for st, (_, sfx, trials) in STAGES.items():
            name = f"r1_{arm}{sfx}_native"
            spec, _ = register.load_run_config(register.RUNS_DIR / f"{name}.yaml")
            yield arm, st, trials, agent, domain, spec


def test_run_configs():
    n = 0
    for arm, st, trials, agent, domain, s in _specs():
        n += 1
        assert (s.agent, s.domain, s.user, s.split, s.eval_final) == (agent, domain, "user_simulator", "train", False)
        assert (s.num_trials, s.seed, s.max_concurrency, s.max_steps, s.max_errors) == (trials, 300, 4, 200, 10)
        assert not s.task_ids and not s.num_tasks  # whole train split = 74 tasks
        assert s.boed_config == "configs/boed_v2.yaml"
        if arm == "plainrepair":
            assert s.agent_kwargs["repair_invalid_user_tools"] is True
        if arm == "teacher":
            register.check_oracle_allowed(s.agent, s.split, s.eval_final, s.task_ids)
    assert n == 8
    assert len(runner.split_task_ids("train")) == 74


@pytest.mark.parametrize("stage", list(STAGES))
def test_study_expansion(stage):
    stem, sfx, trials = STAGES[stage]
    study = runner.load_study(stem)
    assert study.deterministic is False and study.split == "train" and study.users == ("native", "mix")
    assert list(study.arms) == [f"r1_{a}{sfx}_native" for a in ARMS]
    steps = runner.expand(study)
    assert len(steps) == 4 * 7  # native + 6 personas per arm
    assert sum(s.n_expected for s in steps) == 4 * 148 * trials
    for s in steps:  # derived persona configs keep the oracle guard (task ids are train-safe)
        if s.arm.startswith("r1_teacher"):
            spec = register.parse_run_config(__import__("yaml").safe_load(s.derived_text or s.config_path.read_text()), stem=s.config_name)
            register.check_oracle_allowed(spec.agent, spec.split, spec.eval_final, spec.task_ids)
    runner.check_step_names(steps)  # register_all + check_names


def test_cost_table(capsys):
    p = Path(paths.PACKAGE_ROOT) / "scripts" / "r1_estimate.py"
    sp = importlib.util.spec_from_file_location("r1_estimate", p)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    text = m.table()
    with capsys.disabled():
        print("\n" + text)
    assert "pilot" in text and "powered" in text
