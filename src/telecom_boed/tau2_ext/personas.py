"""[telecom_boed] W3-S12: six robustness-protocol user personas as tau2 user classes (REPLACE the task's built-in persona).

tau2 passes `str(task.user_scenario)` to the user as `instructions` (runner/build.py:157-177). That string is
  "Persona:\\n\\t<persona lines>\\nInstructions:\\n\\t..."   (data_model/tasks.py:71-78; no Persona block if persona is None).
`--user-persona`/PersonaConfig only ADD text, so we rewrite `instructions` in __init__: drop the built-in Persona block
(everything before the first column-0 "Instructions:" line; persona lines are tab-indented so cannot collide) and
insert ours in the same place. The task's own text (reason_for_call, known_info, task_instructions) is untouched.
The stock user stays registered by tau2 as `user_simulator` ("native"): its own condition in every comparison.
"""
from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import Any, Optional

import yaml
from tau2.user.user_simulator import UserSimulator

from telecom_boed import paths

NATIVE = "user_simulator"  # tau2 registry name of the stock user (registry.py:283)
PERSONAS: tuple[str, ...] = ("cooperative", "terse", "confused", "contradictory", "noncompliant", "delayed")
PERSONA_DIR = paths.CONFIG_DIR / "personas"
_INSTR_MARK = "\nInstructions:\n"


def user_name(persona: str) -> str:
    """'native' -> tau2 default user; otherwise 'boed-user-<persona>'."""
    if persona == "native":
        return NATIVE
    if persona not in PERSONAS:
        raise ValueError(f"unknown persona {persona!r}; expected 'native' or one of {PERSONAS}")
    return f"boed-user-{persona}"


def load_persona(persona: str) -> dict[str, Any]:
    """yaml fields (name, user_model, description, deviation, text) + `sha256` of the yaml bytes."""
    p = PERSONA_DIR / f"{persona}.yaml"
    raw = p.read_bytes()
    cfg = yaml.safe_load(raw)
    assert cfg["name"] == persona, (p, cfg["name"])
    cfg["sha256"] = hashlib.sha256(raw).hexdigest()
    return cfg


def persona_block(text: str) -> str:
    """Same layout as UserScenario.__str__: 'Persona:' then tab-indented text (ends without newline)."""
    return "Persona:\n" + textwrap.indent(text.strip("\n"), "\t")


def replace_persona(instructions: str, text: str) -> str:
    """Remove the built-in Persona block of `str(task.user_scenario)` (if any) and insert ours before 'Instructions:'."""
    if instructions.startswith("Persona:\n"):
        i = instructions.find(_INSTR_MARK)
        if i < 0:
            raise ValueError("built-in Persona block found but no 'Instructions:' line follows it")
        rest = instructions[i + 1:]
    elif instructions.startswith("Instructions:\n"):
        rest = instructions
    else:
        raise ValueError("unexpected user_scenario format (neither 'Persona:' nor 'Instructions:' first)")
    return persona_block(text) + "\n" + rest


class PersonaUser(UserSimulator):
    """Base class: UserSimulator whose instructions carry our persona instead of the built-in one."""

    persona_name: str = ""
    persona_sha256: str = ""
    persona_text: str = ""

    def __init__(self, llm: str, instructions: Optional[str] = None, *args: Any, **kw: Any) -> None:
        if instructions is not None:
            instructions = replace_persona(instructions, self.persona_text)
        super().__init__(llm, instructions, *args, **kw)

    @classmethod
    def run_metadata(cls) -> dict[str, str]:
        return {"persona": cls.persona_name, "persona_yaml_sha256": cls.persona_sha256, "user_name": user_name(cls.persona_name)}


def _make(persona: str) -> type[PersonaUser]:
    cfg = load_persona(persona)
    return type(f"{persona.capitalize()}User", (PersonaUser,), {
        "persona_name": persona, "persona_sha256": cfg["sha256"], "persona_text": cfg["text"].strip(), "__module__": __name__})


USER_CLASSES: dict[str, type[PersonaUser]] = {p: _make(p) for p in PERSONAS}


def register(registry: Any = None) -> list[str]:
    """Register `boed-user-<persona>` x6 in the tau2 registry (default: the global one). Idempotent."""
    if registry is None:
        from tau2.registry import registry
    have = set(registry.get_users())
    names = []
    for p in PERSONAS:
        n = user_name(p)
        if n not in have:
            registry.register_user(USER_CLASSES[p], n)
        names.append(n)
    return names
