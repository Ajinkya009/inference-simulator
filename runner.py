"""Open-loop Poisson arrival driver + lambda sweep.

Open-loop = arrivals are independent of completions (true Poisson). If the
server can't keep up, calls pile up and latency grows super-linearly -- that
pileup is exactly what reveals the knee. A safety cap (max_concurrent_calls)
prevents the client itself from OOMing during collapse.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import List

import httpx

from .call import run_call
from .client import TurnResult
from .config import Config
from .metrics import StepCollector, StepStats
from .prompt import calibrate


def _make_client(cfg: Config, transport=None) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=cfg.max_connections,
                          max_keepalive_connections=cfg.max_connections)
    timeout = httpx.Timeout(cfg.request_timeout_s, connect=cfg.connect_timeout_s)
    headers = {"Authorization": f"Bearer {cfg.api_key}"}
    return httpx.AsyncClient(base_url=cfg.base_url, limits=limits,
                             timeout=timeout, headers=headers,
                             transport=transport)


async def _run_step(cfg: Config, client: httpx.AsyncClient, lam: float,
                    chars_per_token: float, rng: random.Random) -> StepStats:
    collector = StepCollector(lam)
    inflight: set = set()
    t_step_start = time.perf_counter()
    warmup_until = t_step_start + cfg.warmup_s
    step_until = t_step_start + cfg.step_duration_s
    call_idx = 0
    measured_window_start = None

    def emit(ci: int, ti: int, offset: float, res: TurnResult):
        # Only keep samples that LANDED inside the steady-state window.
        nonlocal measured_window_start
        now = time.perf_counter()
        if now < warmup_until:
            return
        if measured_window_start is None:
            measured_window_start = now
        collector.add(res)

    # Poisson arrivals: exponential inter-arrival with mean 1/lam.
    while time.perf_counter() < step_until:
        if len(inflight) < cfg.max_concurrent_calls:
            task = asyncio.create_task(
                run_call(call_idx, client, cfg, chars_per_token,
                         rng, t_step_start, emit))
            inflight.add(task)
            task.add_done_callback(inflight.discard)
            call_idx += 1
        # next arrival
        await asyncio.sleep(rng.expovariate(lam))

    # Drain remaining in-flight calls (bounded by cooldown).
    if inflight:
        try:
            await asyncio.wait(inflight, timeout=cfg.cooldown_s)
        except Exception:  # noqa: BLE001
            pass
    for t in list(inflight):
        t.cancel()

    window = (time.perf_counter() - measured_window_start
              if measured_window_start else cfg.step_duration_s - cfg.warmup_s)
    stats = collector.finalize(max(window, 1e-3))
    return stats


async def run_sweep(cfg: Config, progress=print, transport=None) -> List[StepStats]:
    rng = random.Random(cfg.seed)
    results: List[StepStats] = []
    async with _make_client(cfg, transport=transport) as client:
        progress(f"[{cfg.label}] calibrating tokenizer against {cfg.base_url} ...")
        cpt = await calibrate(client, cfg.model, rng)
        progress(f"[{cfg.label}] chars/token = {cpt:.3f} "
                 f"(target system prompt = {cfg.system_prompt_tokens} tok)")

        for lam in cfg.arrival_rates:
            progress(f"[{cfg.label}] lambda={lam:.3f} calls/s  "
                     f"(window {cfg.step_duration_s:.0f}s, warmup {cfg.warmup_s:.0f}s) ...")
            stats = await _run_step(cfg, client, lam, cpt, rng)
            results.append(stats)
            progress(f"    -> turns={stats.n_turns} err={stats.error_rate:.1%} "
                     f"TTFT p95={stats.ttft_ms['p95']:.0f}ms "
                     f"ITL p95={stats.itl_ms['p95']:.1f}ms "
                     f"cache_hit={stats.cache_hit_rate:.0%} "
                     f"conc~{stats.achieved_concurrency:.1f}")
            await asyncio.sleep(cfg.cooldown_s)
    return results
