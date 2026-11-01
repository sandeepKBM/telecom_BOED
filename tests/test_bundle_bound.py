"""A REQUEST_DEVICE candidate with a huge action bundle (e.g. the model echoing its system prompt, which names every phone tool: 31
parsed actions in job 60455429) must not make the filter enumerate 2*3^n branch combinations: that blew the tau2 client to 53 GB."""
import resource
import subprocess
import sys
import textwrap

import pytest

from telecom_boed.belief.filter import ParticleFilter
try:
    from telecom_boed.belief.filter import MAX_COMBOS
except ImportError:
    MAX_COMBOS = 20_000
from telecom_boed.contracts import CandidateAction, DecisionKind, DeviceAction
from telecom_boed.belief.user_model import UserModel  # noqa: F401  (import check)

TOOLS = ["check_network_status", "check_status_bar", "reboot_device", "check_vpn_status", "check_sim_status", "run_speed_test",
         "check_apn_settings", "check_wifi_status", "toggle_data", "toggle_roaming", "toggle_airplane_mode", "check_app_status"]


def _cand(n):
    return CandidateAction(DecisionKind.REQUEST_DEVICE, tuple(DeviceAction("user", TOOLS[i % len(TOOLS)], ()) for i in range(n)))


def _flt():
    from telecom_boed.tau2_ext.agents import _load_cfg
    from telecom_boed import paths
    return ParticleFilter("deployable", _load_cfg(str(paths.CONFIG_DIR / "boed_default.yaml")), "default")


def test_small_bundle_still_enumerated():
    f = _flt()
    combos, _ = f._combos(_cand(3))
    assert len(combos) <= MAX_COMBOS and len(combos) > 3


def test_huge_bundle_raises_not_enumerates():
    f = _flt()
    with pytest.raises(ValueError, match="bundle"):
        f._combos(_cand(31))


def test_huge_bundle_memory_bounded():
    """Child process with a 6 GB address-space cap: before the fix this dies with MemoryError/killed, after it raises ValueError."""
    code = textwrap.dedent("""
        import resource, sys
        resource.setrlimit(resource.RLIMIT_AS, (6_000_000_000, 6_000_000_000))
        sys.path.insert(0, %r)
        import conftest
        from test_bundle_bound import _cand, _flt
        try:
            _flt()._combos(_cand(31))
        except ValueError:
            print("OK-VALUEERROR")
        except MemoryError:
            print("MEMORYERROR")
    """ % str(__import__("pathlib").Path(__file__).parent))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "OMP_NUM_THREADS": "1"})
    assert "OK-VALUEERROR" in r.stdout, (r.stdout[-300:], r.stderr[-300:])
