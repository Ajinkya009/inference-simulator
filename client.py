"""Streaming client + per-request latency capture.

TTFT  = wall time from just-before-send to the first non-empty content delta
        (includes server queueing -- exactly what production feels).
ITL   = gaps between consecutive output tokens (one sample per token gap).
TPOT  = (last_token_time - first_token_time) / (n_tokens - 1).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx


@dataclass
class TurnResult:
    ok: bool
    ttft_ms: float = 0.0
    itls_ms: List[float] = field(default_factory=list)   # per-token gaps
    n_output: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    total_s: float = 0.0
    error: Optional[str] = None

    @property
    def tpot_ms(self) -> float:
        if self.n_output <= 1:
            return 0.0
        # mean of inter-token gaps == TPOT
        return sum(self.itls_ms) / len(self.itls_ms) if self.itls_ms else 0.0


async def stream_chat(client: httpx.AsyncClient, model: str,
                      messages: list, max_tokens: int,
                      ignore_eos: bool, temperature: float) -> TurnResult:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        # SGLang extension: forces exactly max_tokens output for clean ITL.
        payload["ignore_eos"] = True

    t_start = time.perf_counter()
    first_tok_t: Optional[float] = None
    last_tok_t: Optional[float] = None
    itls: List[float] = []
    n_out = 0
    usage = {}

    try:
        async with client.stream("POST", "/chat/completions",
                                 json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "ignore")[:200]
                return TurnResult(ok=False, error=f"HTTP {resp.status_code}: {body}",
                                  total_s=time.perf_counter() - t_start)
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue

                now = time.perf_counter()
                if first_tok_t is None:
                    first_tok_t = now
                else:
                    itls.append((now - last_tok_t) * 1000.0)
                last_tok_t = now
                n_out += 1
    except (httpx.HTTPError, httpx.StreamError) as e:
        return TurnResult(ok=False, error=str(e),
                          total_s=time.perf_counter() - t_start)

    if first_tok_t is None:
        return TurnResult(ok=False, error="no tokens streamed",
                          total_s=time.perf_counter() - t_start)

    from prompt import measured_prompt_tokens, cached_prompt_tokens
    return TurnResult(
        ok=True,
        ttft_ms=(first_tok_t - t_start) * 1000.0,
        itls_ms=itls,
        n_output=n_out,
        prompt_tokens=measured_prompt_tokens(usage),
        cached_tokens=cached_prompt_tokens(usage),
        total_s=time.perf_counter() - t_start,
    )
