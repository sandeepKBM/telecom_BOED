"""[telecom_boed] W3-S12 tests: persona user classes replace the built-in persona; native untouched. TRAIN tasks only."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml
from tau2.data_model.tasks import Task
from tau2.registry import RegistryInfo, registry
from tau2.user.user_simulator import UserSimulator

from telecom_boed import paths
from telecom_boed.tau2_ext import personas as P

FIX = Path(__file__).parent / "fixtures" / "W3-S12"


def _pick(catalog: list[dict], train_ids: list[str]) -> dict[str, Task]:
    """One TRAIN task per built-in persona: None / Easy / Hard."""
    by_id = {t["id"]: t for t in catalog}
    out: dict[str, Task] = {}
    for tid in train_ids:
        tag = tid.rsplit("[PERSONA:", 1)[1].rstrip("]")
        if tag not in out:
            out[tag] = Task.model_validate(by_id[tid])
    assert set(out) == {"None", "Easy", "Hard"}, set(out)
    return out


@pytest.fixture(scope="module")
def picked(catalog, train_ids) -> dict[str, Task]:
    return _pick(catalog, train_ids)


def test_builtin_persona_removed_and_ours_inserted(picked):
    # built-in persona text exists for Easy/Hard only; None has no block (ours is simply inserted)
    assert picked["None"].user_scenario.persona is None
    for tag in ("Easy", "Hard"):
        assert picked[tag].user_scenario.persona
    manifest = {}
    for tag, task in picked.items():
        sc = task.user_scenario
        ins = sc.instructions
        for persona in P.PERSONAS:
            cls = P.USER_CLASSES[persona]
            u = cls(llm="dummy-llm", instructions=str(sc))
            sp = u.system_prompt
            ours = cls.persona_text
            assert ours and ours in sp.replace("\n\t", "\n")
            if sc.persona:
                assert textwrap.indent(sc.persona, "\t") not in sp
                for line in sc.persona.strip().splitlines():
                    if len(line.strip()) > 25:
                        assert line.strip() not in sp
            # task content verbatim
            for field in ("reason_for_call", "known_info", "task_instructions"):
                val = getattr(ins, field)
                if val:
                    assert val.strip().splitlines()[0] in sp
                    assert textwrap.indent(textwrap.indent(val, "\t"), "\t") in sp  # nested under 'Instructions:'
            assert textwrap.indent(str(ins), "\t") in sp  # the whole instructions block, byte-identical
            # persona block sits immediately before Instructions:
            assert sp.index("Persona:") < sp.index("Instructions:") < sp.index("</scenario>")
            assert sp.count("Persona:\n") == 1
            assert cls.run_metadata()["persona_yaml_sha256"] == cls.persona_sha256
            manifest[f"{tag}/{persona}"] = len(sp)
    FIX.mkdir(parents=True, exist_ok=True)
    (FIX / "prompt_lengths.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))


def test_native_untouched(picked):
    assert P.user_name("native") == "user_simulator"
    assert registry.get_user_constructor(P.NATIVE) is UserSimulator
    for task in picked.values():
        theirs = UserSimulator(llm="dummy-llm", instructions=str(task.user_scenario)).system_prompt
        built = registry.get_user_constructor(P.user_name("native"))(llm="dummy-llm", instructions=str(task.user_scenario))
        assert built.system_prompt == theirs
        assert theirs.encode() == built.system_prompt.encode()
        if task.user_scenario.persona:
            assert textwrap.indent(task.user_scenario.persona, "\t") in theirs  # stock persona preserved


def test_persona_names_match_user_model_files():
    um = sorted(p.stem for p in (paths.CONFIG_DIR / "user_model").glob("*.yaml") if p.stem != "default")
    ps = sorted(p.stem for p in P.PERSONA_DIR.glob("*.yaml"))
    assert um == ps == sorted(P.PERSONAS)
    for stem in ps:
        cfg = P.load_persona(stem)
        assert yaml.safe_load((paths.CONFIG_DIR / "user_model" / f"{stem}.yaml").read_text())["name"] == stem
        assert cfg["user_model"] == stem and cfg["name"] == stem
        assert all(cfg[k] for k in ("description", "deviation", "text"))
        assert len(cfg["text"].split()) <= 150
        assert len(cfg["sha256"]) == 64
    assert P.user_name("terse") == "boed-user-terse"
    with pytest.raises(ValueError):
        P.user_name("bogus")


def test_register_idempotent():
    a = P.register(registry)
    b = P.register(registry)
    assert a == b == [f"boed-user-{p}" for p in P.PERSONAS]
    for p in P.PERSONAS:
        assert registry.get_user_constructor(P.user_name(p)) is P.USER_CLASSES[p]
    assert set(a) <= set(registry.get_users())
