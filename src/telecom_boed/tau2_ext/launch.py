"""[telecom_boed] Patched entry point for every tau2 run (minimal version Opus 2026-10-16; W3-S10 added `--config`).

    python -m telecom_boed.tau2_ext.launch [--eval-final] --tau2-cli -- run --domain telecom ...   # raw tau2 CLI args
    python -m telecom_boed.tau2_ext.launch --config configs/runs/<name>.yaml                       # named run config

`--config` mode (schema + side channel: tau2_ext/register.py): load+validate the run config, run the same guards and
patches, `register.register_all()` (our agents/users), build tau2's `TextRunConfig` (data_model/simulation.py:561) and
call `tau2.runner.batch.run_domain` (batch.py:1085) in-process. save_to = $TAU2_SAVE_TO or the config name; refuses to
overwrite an existing results.json (set TAU2_RESUME=1 to resume). run_meta.json also holds the config text + sha256.

What it guarantees, in order, BEFORE tau2 is imported:
1. TAU2_DATA_DIR is set and lies outside the pinned clone (tau2 reads it once at import; results go there).
2. Split guard: `--task-split-name test` (test-40) only with --eval-final.
3. In-process only: `--workers` must be absent or 0 (registrations and patches don't reach worker processes).
4. The process patches (env/tau2_patches.py: per-instance VPN default, deterministic bill ids) are applied, so
   every condition (including the plain baseline) runs on the same patched benchmark. This is a stated deviation
   from upstream: the unpatched VPN default leaks across concurrent episodes.
5. run_meta.json (argv, patches, pins, md5s, versions, host) is written to $TELECOM_RUN_DIR (or the cwd).
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from telecom_boed import contracts, paths
from telecom_boed.tau2_ext import register


def _guard(tau2_args: list[str], eval_final: bool) -> None:
    data = os.environ.get("TAU2_DATA_DIR")
    if not data:
        raise SystemExit("launch: TAU2_DATA_DIR is unset (tau2 would write into the clone)")
    if paths.tau2_root().resolve() in Path(data).resolve().parents or Path(data).resolve() == paths.tau2_root().resolve():
        raise SystemExit(f"launch: TAU2_DATA_DIR={data} is inside the pinned clone")
    if "tau2.utils.utils" in sys.modules:  # in-process callers (tests) may import tau2 first, but only with the same data dir
        cur = getattr(sys.modules["tau2.utils.utils"], "DATA_DIR", None)
        if cur is None or Path(cur).resolve() != Path(data).resolve():
            raise SystemExit("launch: tau2 was imported before the guards ran (DATA_DIR differs from TAU2_DATA_DIR)")
    if "--task-split-name" in tau2_args:
        split = tau2_args[tau2_args.index("--task-split-name") + 1]
        if split == "test" and not eval_final:
            raise SystemExit("launch: the test split (test-40) requires --eval-final")
    if "--workers" in tau2_args and tau2_args[tau2_args.index("--workers") + 1] not in ("0",):
        raise SystemExit("launch: --workers must be 0 (in-process patches/registrations)")


def _run_meta(tau2_args: list[str], patches: list[str], extra: dict | None = None) -> dict:
    tasks, split = paths.telecom_catalog_paths()

    def ver(p: str) -> str:
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return "absent"

    meta = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "tau2_args": tau2_args,
        "patches": patches,
        "tau2_sha": paths.TAU2_SHA,
        "tasks_json_md5": paths.md5(tasks),
        "split_tasks_json_md5": paths.md5(split),
        "contract_version": contracts.CONTRACT_VERSION,
        "versions": {p: ver(p) for p in ("tau2", "litellm", "pydantic", "telecom_boed")},
        "tau2_data_dir": os.environ["TAU2_DATA_DIR"],
    }
    meta.update(extra or {})
    return meta


def _write_meta(meta: dict) -> Path:
    out = Path(os.environ.get("TELECOM_RUN_DIR", "."))
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return out / "run_meta.json"


# ------------------------------------------------------------------------------------------ --config mode
def _guard_config(spec: register.RunSpec) -> None:
    """Same guards as the raw-CLI mode, driven by the config (the schema enforces them too; this is defence in depth)."""
    args = ["--task-split-name", spec.split, "--workers", "0"]
    _guard(args, spec.eval_final)
    if spec.split in register.TEST_BEARING_SPLITS and not spec.eval_final and not (spec.task_ids and register.train_safe(spec.task_ids)):
        raise SystemExit(f"launch: split {spec.split!r} contains test-40 and requires eval_final: true")


def build_text_run_config(spec: register.RunSpec, save_to: str, auto_resume: bool = False) -> Any:
    """tau2 TextRunConfig (data_model/simulation.py:561; base fields :296-:430, text fields :569-:606)."""
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain=spec.domain,  # :303
        task_split_name=spec.split,  # :317 (task_set_name stays None -> domain, batch.py:1050)
        task_ids=list(spec.task_ids) if spec.task_ids else None,  # :324
        num_tasks=spec.num_tasks,  # :331
        llm_user=(spec.user_llm or spec.llm).model,  # :340
        llm_args_user=register.llm_args_for(spec, "user"),  # :347
        num_trials=spec.num_trials,  # :356
        max_errors=spec.max_errors,  # :363
        save_to=save_to,  # :377 -> DATA_DIR/simulations/<save_to>/results.json (batch.py:1079)
        max_concurrency=spec.max_concurrency,  # :384
        workers=0,  # :391 in-process
        seed=spec.seed,  # :410
        agent=spec.agent,  # :569
        llm_agent=spec.llm.model,  # :576
        llm_args_agent=register.llm_args_for(spec, "agent"),  # :583
        user=spec.user,  # :592
        max_steps=spec.max_steps,  # :601
        auto_resume=auto_resume,
    )


def run_config(config_path: str | Path) -> Any:
    """Run one named config in-process; returns tau2's Results. Raises SystemExit on guard failures."""
    try:
        spec, text = register.load_run_config(config_path)
    except register.ConfigError as e:
        raise SystemExit(f"launch: bad run config {config_path}: {e}") from e
    _guard_config(spec)
    paths.assert_catalog_pins()
    from telecom_boed.env import tau2_patches  # imports tau2 (after the data-dir guard)

    tau2_patches.apply()
    names = register.register_all()
    try:
        register.check_names(spec)
    except register.ConfigError as e:
        raise SystemExit(f"launch: {e}") from e
    register.set_run(spec)
    save_to = os.environ.get("TAU2_SAVE_TO") or spec.name
    results = Path(os.environ["TAU2_DATA_DIR"]) / "simulations" / save_to / "results.json"
    resume = os.environ.get("TAU2_RESUME") == "1"
    if results.exists() and not resume:
        raise SystemExit(f"launch: {results} exists (immutable runs; use a new TAU2_SAVE_TO or TAU2_RESUME=1)")
    bc = register.resolve_boed_config(spec.boed_config)
    extra = {
        "mode": "config", "save_to": save_to, "results_path": str(results),
        "run_config": {"path": str(config_path), "name": spec.name, "sha256": register.sha256_text(text), "content": text,
                       "spec": spec.to_dict()},
        "boed_config": {"path": str(bc), "sha256": register.sha256_text(bc.read_text())},
        "registered_by_us": names, "registry": register.registered_names(),
    }
    _write_meta(_run_meta(["--config", str(config_path)], list(tau2_patches.PATCHES), extra))
    from tau2.runner.batch import run_domain

    try:
        text_cfg = build_text_run_config(spec, save_to, auto_resume=resume)
    except register.ConfigError as e:
        raise SystemExit(f"launch: {e}") from e
    return run_domain(text_cfg)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" in argv and "--" not in argv:
        ap = argparse.ArgumentParser(prog="telecom_boed.tau2_ext.launch")
        ap.add_argument("--config", required=True, help="run config yaml (name under configs/runs/ or a path)")
        res = run_config(ap.parse_args(argv).config)
        print(f"launch: done, {len(res.simulations)} simulations -> {os.environ['TAU2_DATA_DIR']}/simulations/")
        return
    if "--" not in argv:
        raise SystemExit("usage: launch [--eval-final] --tau2-cli -- <tau2 args> | launch --config <run.yaml>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser(prog="telecom_boed.tau2_ext.launch")
    ap.add_argument("--tau2-cli", action="store_true", required=True, help="pass the remaining args to the tau2 CLI")
    ap.add_argument("--eval-final", action="store_true", help="allow the test split (test-40)")
    opts = ap.parse_args(argv[:cut])
    tau2_args = argv[cut + 1:]
    _guard(tau2_args, opts.eval_final)
    paths.assert_catalog_pins()
    from telecom_boed.env import tau2_patches  # imports tau2 (after the data-dir guard)

    tau2_patches.apply()
    _write_meta(_run_meta(tau2_args, list(tau2_patches.PATCHES)))
    from tau2 import cli

    sys.argv = ["tau2", *tau2_args]
    cli.main()


if __name__ == "__main__":
    main()
