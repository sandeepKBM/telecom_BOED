"""P0-4: hardened launcher copies (syntax, dry-run, required hardening, run_client recycling with a fake client). CPU only."""
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SLURM = Path(__file__).resolve().parents[1] / "slurm"
V2 = ["local_tau2_vllm_v2.sh", "tau2_launch_l2_v2.sbatch"]


def test_syntax_and_dry_run():
    for f in V2:
        assert subprocess.run(["bash", "-n", str(SLURM / f)]).returncode == 0
        r = subprocess.run(["bash", str(SLURM / f)], env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
        assert r.returncode == 0 and "STARTUP_RETRIES=2" in r.stdout and "-X faulthandler" in r.stdout, r.stderr


def test_hardening_present_and_originals_untouched():
    sb = (SLURM / "tau2_launch_l2_v2.sbatch").read_text()
    sh = (SLURM / "local_tau2_vllm_v2.sh").read_text()
    assert "#SBATCH --exclude=nid003768" in sb
    for t in (sb, sh):
        assert "-X faulthandler" in t and "PYTHONFAULTHANDLER=1" in t
        assert "FC-013" in t and "Loading safetensors checkpoint" in t  # retry only before weight load
        assert "FC-011" in t and "rss.log" in t and "run_client" in t
    assert "TELECOM_LAUNCHER_COPY" in sh and "exec bash" in sh  # self-snapshot kept
    for orig in ("tau2_launch_l2.sbatch", "local_tau2_vllm.sh"):
        o = (SLURM / orig).read_text()
        assert "exclude=nid003768" not in o and "run_client" not in o  # originals are unmodified


def _funcs(text):
    m = re.search(r"# >>> client_recycle.*?# <<< client_recycle", text, re.S)
    assert m
    return m.group(0)


def test_run_client_recycles_on_rss_and_resumes(tmp_path):
    assert _funcs((SLURM / "tau2_launch_l2_v2.sbatch").read_text()) == _funcs((SLURM / "local_tau2_vllm_v2.sh").read_text())
    client = tmp_path / "fake_client.py"
    state = tmp_path / "state"
    client.write_text(textwrap.dedent(f"""
        import os, sys, time
        n = len(open({str(state)!r}).read()) if os.path.exists({str(state)!r}) else 0
        open({str(state)!r}, "a").write("x")
        print("start", n, "resume=" + os.environ.get("TAU2_RESUME", ""), flush=True)
        if n == 0:                       # first start: leak ~200 MB then hang -> must be recycled
            blob = bytearray(200 * 1024 * 1024); time.sleep(60)
        print("done", flush=True)        # restarted: finishes normally
    """))
    script = tmp_path / "t.sh"
    script.write_text(_funcs((SLURM / "local_tau2_vllm_v2.sh").read_text()) + textwrap.dedent(f"""
        export RSS_CAP_KB=100000 RECYCLE_POLL_S=1 RECYCLE_GRACE_S=1 MAX_RECYCLES=2 RSS_LOG={tmp_path}/rss.log
        run_client {tmp_path}/c.log 60 {sys.executable} {client}
        echo "RC=$RC RECYCLES=$RECYCLES RESUME_AFTER=${{TAU2_RESUME:-unset}}"
    """))
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=90)
    assert "RC=0 RECYCLES=1 RESUME_AFTER=unset" in r.stdout, (r.stdout, r.stderr)
    log = (tmp_path / "c.log").read_text()
    assert "start 0 resume=\n" in log and "start 1 resume=1" in log and "done" in log
    assert "recycle client" in (tmp_path / "rss.log").read_text()


def test_run_client_no_recycle_when_disabled(tmp_path):
    script = tmp_path / "t.sh"
    script.write_text(_funcs((SLURM / "local_tau2_vllm_v2.sh").read_text()) + textwrap.dedent(f"""
        export MAX_RECYCLES=0 RECYCLE_POLL_S=1
        run_client {tmp_path}/c.log 30 bash -c 'exit 7'
        echo "RC=$RC RECYCLES=$RECYCLES"
    """))
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
    assert "RC=7 RECYCLES=0" in r.stdout, (r.stdout, r.stderr)
