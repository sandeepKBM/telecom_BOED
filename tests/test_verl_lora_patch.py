"""CPU check of patches/verl/per_request_lora.patch (use_lora flag). Skips when the patched verl is not importable."""
import asyncio
import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("ray")
vas = pytest.importorskip("verl.workers.rollout.vllm_rollout.vllm_async_server")
lls = pytest.importorskip("verl.workers.rollout.llm_server")
if "use_lora" not in inspect.signature(vas.vLLMHttpServer.generate).parameters:
    pytest.skip("verl is not the ddl-patched copy (no use_lora)", allow_module_level=True)

from verl.workers.rollout.replica import TokenOutput  # noqa: E402


class _CfgDict(SimpleNamespace):
    def get(self, k, d=None):
        return getattr(self, k, d)


class _FakeEngine:
    def __init__(self):
        self.calls = []

    async def list_loras(self):
        return {vas.VLLM_LORA_INT_ID}

    async def generate(self, **kw):
        self.calls.append(kw)
        yield SimpleNamespace(outputs=[])  # empty outputs -> server returns an "aborted" TokenOutput early


class _Srv(vas.vLLMHttpServer):
    lora_as_adapter = True  # shadow the property: adapter is loaded


def _server(lora_as_adapter=True):
    s = object.__new__(_Srv)
    s.lora_as_adapter = lora_as_adapter
    s._disaggregation_role = None
    s._pd_decode_peers = None
    s.replica_rank = 0
    s.config = _CfgDict(max_model_len=4096, response_length=64, prompt_length=64, full_determinism=False,
                        replica_rank=0, seed=0, enable_rollout_routing_replay=False, mtp=None)
    s.model_config = SimpleNamespace(processor=None)
    s.engine = _FakeEngine()
    s._submission_paused = False
    s._rejecting = False
    s._admitting = 0
    s.global_steps = 0
    return s


def _gen(s, **kw):
    return asyncio.run(s.generate(prompt_ids=[1, 2, 3], sampling_params={}, request_id="r", **kw))


def test_server_default_keeps_adapter():
    s = _server()
    _gen(s)
    lr = s.engine.calls[0]["lora_request"]
    assert lr is not None and lr.lora_int_id == vas.VLLM_LORA_INT_ID


def test_server_use_lora_false_sends_none():
    s = _server()
    _gen(s, use_lora=False)
    assert s.engine.calls[0]["lora_request"] is None


def test_server_without_lora_unchanged():
    s = _server(lora_as_adapter=False)
    _gen(s)
    assert s.engine.calls[0]["lora_request"] is None


def _client():
    c = object.__new__(lls.LLMServerClient)
    c.config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(name="vllm", full_determinism=False)))
    sent = []

    class _Remote:
        async def remote(self, **kw):
            sent.append(kw)
            return TokenOutput(token_ids=[], log_probs=None, routed_experts=None, stop_reason="aborted",
                               extra_fields={"global_steps": 0})

    async def acquire(request_id, **extra):
        return "s0", SimpleNamespace(generate=_Remote())

    c._acquire_server = acquire
    c._release_server = lambda *a, **k: None
    return c, sent


def test_client_forwards_flag_only_when_false():
    c, sent = _client()
    asyncio.run(c.generate("r", prompt_ids=[1], sampling_params={}))
    asyncio.run(c.generate("r", prompt_ids=[1], sampling_params={}, use_lora=False))
    assert "use_lora" not in sent[0]  # default call identical to upstream
    assert sent[1]["use_lora"] is False
