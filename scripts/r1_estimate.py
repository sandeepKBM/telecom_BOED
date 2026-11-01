#!/usr/bin/env python3
"""[telecom_boed] P2 (R1) cost table, machine-computed from the study YAMLs (no hand-typed numbers).
Episodes per stage = runner.expand() step n_expected; GPU-h = episodes / theta (episodes per GPU-h, plan range 130/150; P1 re-measures).
Perlmutter packing: slurm/tau2_launch_l2_v2.sbatch runs the CONFIGS list sequentially behind ONE vLLM start, so a job = greedy run of
steps (in study order) whose estimated wall (startup + episodes/theta*margin) fits WALL_CAP_H. Planning knobs below are assumptions."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from telecom_boed.eval import runner  # noqa: E402

STAGES = {"pilot": "r1_train", "powered": "r1_train_powered"}
THETAS = (130.0, 150.0)
STARTUP_H = 0.1      # vLLM start + health (assumption; l2 jobs show ~3-10 min)
MARGIN = 1.25        # wall safety factor on the episode time (assumption)
WALL_CAP_H = 4.0     # target sbatch -t per job at the slow theta (assumption; not a QOS limit)


def stage_steps(stage: str):
    study = runner.load_study(STAGES[stage])
    return study, runner.expand(study)


def pack(steps, theta: float, cap_h: float = WALL_CAP_H):
    jobs, cur, wall = [], [], STARTUP_H
    for s in steps:
        w = s.n_expected / theta * MARGIN
        if cur and wall + w > cap_h:
            jobs.append((cur, wall)); cur, wall = [], STARTUP_H
        cur.append(s); wall += w
    if cur:
        jobs.append((cur, wall))
    return jobs


def table() -> str:
    out = [f"{'stage':<8}{'steps':>6}{'episodes':>9}" + "".join(f"{'GPU-h@' + format(t, 'g'):>12}" for t in THETAS) + "   jobs@130 (<=" + f"{WALL_CAP_H:g}h wall, x{MARGIN} margin)"]
    for st in STAGES:
        _, steps = stage_steps(st)
        eps = sum(s.n_expected for s in steps)
        jobs = pack(steps, THETAS[0])
        out.append(f"{st:<8}{len(steps):>6}{eps:>9}" + "".join(f"{eps / t:>12.2f}" for t in THETAS) + f"   {len(jobs)}")
    tot = sum(sum(s.n_expected for s in stage_steps(st)[1]) for st in STAGES)
    out.append(f"{'total':<8}{'':>6}{tot:>9}" + "".join(f"{tot / t:>12.2f}" for t in THETAS))
    out.append("(oracle-teacher steps make no agent LLM calls; counted at full theta, so these are upper bounds for that arm)")
    for st in STAGES:
        _, steps = stage_steps(st)
        for i, (j, w) in enumerate(pack(steps, THETAS[0]), 1):
            arms = sorted({s.arm for s in j})
            out.append(f"  {st} job {i}: {len(j)} configs, {sum(s.n_expected for s in j)} eps, est wall {w:.2f} h, arms {arms}")
    return "\n".join(out)


if __name__ == "__main__":
    print(table())
