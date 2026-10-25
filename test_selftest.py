"""In-process end-to-end self-test (no ports, real streaming timing).

A custom httpx transport simulates an SGLang endpoint: load-dependent TTFT,
per-token ITL sleeps, and usage with cached_tokens. This exercises the real
TTFT/ITL capture path, Poisson driver, metrics, and knee detection.
"""
import asyncio, json, time
import httpx

from config import Config
from runner import run_sweep
from metrics import slo_knee, kneedle_knee, steps_to_rows
from prompt import PromptFactory, TokenCounter

TTFT_BASE = 0.035
KNEE_CONC = 8
SLOPE = 0.022
ITL = 0.009

class MockStream(httpx.AsyncByteStream):
    def __init__(self, items, on_close):
        self._items = items
        self._on_close = on_close
    async def __aiter__(self):
        for delay, data in self._items:
            if delay:
                await asyncio.sleep(delay)
            yield data
    async def aclose(self):
        self._on_close()

class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self):
        self.inflight = 0
    async def handle_async_request(self, request):
        body = json.loads(request.content or b"{}")
        msgs = body.get("messages", [])
        max_tokens = int(body.get("max_tokens", 16))
        stream = bool(body.get("stream", False))
        prompt_chars = sum(len(m.get("content", "")) for m in msgs)
        ptok = max(1, prompt_chars // 4)
        has_sys = any(m.get("role") == "system" for m in msgs)
        cached = max(0, ptok - 60) if has_sys else 0

        self.inflight += 1
        conc = self.inflight
        ttft = TTFT_BASE + SLOPE * max(0, conc - KNEE_CONC)

        def on_close():
            self.inflight -= 1

        usage = {"prompt_tokens": ptok, "completion_tokens": max_tokens,
                 "total_tokens": ptok + max_tokens,
                 "prompt_tokens_details": {"cached_tokens": cached}}

        if not stream:
            on_close()
            data = json.dumps({"choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "length"}], "usage": usage}).encode()
            return httpx.Response(200, headers={"content-type": "application/json"},
                                  content=data)

        def sse(obj):
            return f"data: {json.dumps(obj)}\n\n".encode()
        items = []
        items.append((ttft, sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})))
        items.append((0.0, sse({"choices": [{"index": 0, "delta": {"content": "tok "}}]})))
        for _ in range(max_tokens - 1):
            items.append((ITL, sse({"choices": [{"index": 0, "delta": {"content": "tok "}}]})))
        items.append((0.0, sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": usage})))
        items.append((0.0, b"data: [DONE]\n\n"))
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              stream=MockStream(items, on_close))

async def main():
    # Unit-level invariants that do not require a tokenizer download or GPU.
    production = Config(prompt_sharing="shared", shared_prefix_tokens=120,
                        customer_context_tokens=20, user_tokens=10, prompt_groups=2, seed=11)
    factory = PromptFactory(production, TokenCounter(), __import__("random").Random(11))
    factory.validate_shared_prefix()
    assert factory.system_prefix(0) == factory.system_prefix(0)
    assert factory.system_prefix(0) != factory.system_prefix(1)
    assert len(factory.customer_context(__import__("random").Random(1)).split()) == 20
    assert len(factory.user(__import__("random").Random(2)).split()) == 10
    cfg = Config(
        base_url="http://mock/v1", label="MOCK",
        system_prompt_tokens=200, inter_turn_gap_s=(0.01, 0.02),
        turns=4, output_tokens=8, output_tokens_min=8, output_tokens_max=8,
        arrival_rates=[2.0, 6.0, 12.0],
        step_duration_s=1.0, warmup_s=0.2, cooldown_s=0.0, drain_s=0.5,
        ttft_slo_ms=120.0, seed=7,
    )
    steps = await run_sweep(cfg, transport=MockTransport())
    print("\nlambda  turns  err%  TTFTp50  TTFTp95  ITLp95  cache%  conc")
    for s in steps:
        print(f"{s.arrival_rate:5.1f}  {s.n_turns:5d}  {s.error_rate*100:4.0f}  "
              f"{s.ttft_ms['p50']:7.0f}  {s.ttft_ms['p95']:7.0f}  "
              f"{s.itl_ms['p95']:6.1f}  {s.cache_hit_rate*100:5.0f}  "
              f"{s.achieved_concurrency:4.1f}")
    sk, kk = slo_knee(steps, cfg.ttft_slo_ms), kneedle_knee(steps)
    print(f"\nSLO knee={sk}  curvature knee={kk}")
    assert any(s.n_turns > 0 for s in steps), "no turns measured"
    assert any(s.cache_hit_rate > 0.5 for s in steps), "cache parse broken"
    assert kk is not None, "knee detection failed"
    # A content chunk is not a token: mock delivers one text fragment per
    # token here, but client keeps exact text and separately tracks gaps.
    assert any(r["completion_tokens"] == 8 and len(r["chunk_gaps_ms"]) == 7
               for s in steps for r in s.request_records), "stream reconstruction/gaps broken"
    # TTFT should be ~flat then rise: last step p95 >> first step p95
    assert steps[-1].ttft_ms["p95"] > steps[0].ttft_ms["p95"] * 1.5, "no knee shape"
    print("SELF-TEST PASSED")

if __name__ == "__main__":
    asyncio.run(main())
