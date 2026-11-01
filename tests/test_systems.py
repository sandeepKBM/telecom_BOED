"""P0-7: eval/systems.py accounting + run_events optional fields + gpu_sampler parse (CPU, stdlib)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from telecom_boed.eval import systems as S

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_count_and_charge_rules():
    assert S.gpu_count("billing=4,cpu=128,gres/gpu=4,node=1") == 4
    assert S.gpu_count("gres/gpu=4,gres/gpu:a100=4") == 4  # typed + untyped total not double-counted
    assert S.gpu_count("cpu=2,mem=1G") == 0
    # full-node QOS on Perlmutter: 2 nodes x 4 GPU-h per node-h x 0.5 h, even if only 1 gpu/node was requested
    assert S.charged_gpu_h("gres/gpu=2,node=2", 1800, "regular", perlmutter=True) == 4.0
    assert S.charged_gpu_h("gres/gpu=1,node=1", 3600, "shared", perlmutter=True) == 1.0
    assert S.charged_gpu_h("gres/gpu=4,node=1", 3600, "", perlmutter=False) == 4.0
    assert S.charged_gpu_h("cpu=2", 3600, "regular", perlmutter=True) is None
    assert S.charged_gpu_h("gres/gpu=1", None) is None


def test_metrics_from_artifacts(tmp_path):
    p = tmp_path / "gpu_samples.csv"
    p.write_text("2026/10/31 10:00:00.1,0,50,10,1000,100\n2026/10/31 10:00:05.1,0,100,10,1000,100\nbad\n")
    rows = S.read_gpu_samples(p)
    assert len(rows) == 2 and S.actor_utilisation(rows) == 0.75
    sims = [{"reward_info": {"reward": 1.0}}, {"reward_info": {"reward": None}}, {"reward_info": {"reward": 0.0}}]
    assert S.verified_count(sims) == 2 and S.verified_per_gpu_h(2, 0.5) == 4.0
    assert S.tokens_per_s({"vllm:generation_tokens_total": 1000.0}, 10)["gen_tok_s"] == 100.0
    ev = [{"item": "a", "enqueue": 10.0}, {"item": "a", "dequeue": 12.5}, {"phase": "consume", "policy_version": 3, "learner_version": 5}]
    assert S.queue_delay(ev)["mean"] == 2.5 and S.freshness_lag(ev)["mean"] == 2.0
    assert S.time_to_quality([(10, 0.1), (20, 0.5)], 0.4) == 20 and S.time_to_quality([(10, 0.1)], 0.4) is None
    out = S.run_systems_metrics()
    assert out["queue_delay_s"] is None and out["actor_utilisation"] is None  # missing inputs -> None, never a guess


def test_run_events_optional_fields(tmp_path):
    out = tmp_path / "e.jsonl"
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "run_events.py"), "emit", "--out", str(out), "--run", "r"]
    subprocess.run(cmd + ["--phase", "old", "k=1"], check=True)
    subprocess.run(cmd + ["--phase", "new", "policy_version=2", "gen_start=now", "enqueue=12.5", "item=x"], check=True)
    old, new = [json.loads(x) for x in out.read_text().splitlines()]
    assert set(old) == {"t", "run", "phase", "k"}  # backward compatible
    assert new["policy_version"] == 2 and new["enqueue"] == 12.5 and isinstance(new["gen_start"], float)


def test_status_alloc_column(tmp_path):
    spec = importlib.util.spec_from_file_location("ts", ROOT / "scripts" / "telecom_status.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    f = tmp_path / "sacct.txt"
    f.write_text("123456|billing=4,gres/gpu=4,node=1|1800|regular\n")
    rows = [{"run": "t1_123456"}, {"run": "x_9.1"}]
    ts.add_alloc_gpu_h(rows, sacct_file=f)
    assert rows[0]["gpu_h_reserved"] == 2.0 and rows[0]["n_gpus_reserved"] == 4 and rows[1]["gpu_h_reserved"] is None
