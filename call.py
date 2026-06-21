"""A single call = one Poisson arrival.

Lifecycle (production-faithful):
  turn 1: send [system(18K) + user1], decode 1 token, DISCARD (static reply).
          -> warms the unique per-call prefix in SGLang's radix cache.
  gap (5-7s)  # user speaking / ASR
  turn 2: send [system + user1 + static1 + user2], stream 100 tok. MEASURED.
  gap
  turn 3: + static2 ... wait, turns 2..N responses come from the LLM, so the
          assistant content we carry forward is the model's own output.
  ...
Full history is carried forward every turn (input grows by ~user+assistant
each turn). The system prefix stays identical -> warm path on turns 2..N.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Callable, List

import httpx

from .client import stream_chat, TurnResult
from .config import Config
from .prompt import build_system_prompt, build_user_message


async def run_call(call_idx: int, client: httpx.AsyncClient, cfg: Config,
                   chars_per_token: float, rng: random.Random,
                   t_step_start: float,
                   emit: Callable[[int, int, float, TurnResult], None]) -> None:
    """Run one full call. `emit(call_idx, turn_idx, arrival_offset, result)`
    is called for every MEASURED turn (2..N)."""
    system_prompt = build_system_prompt(cfg.system_prompt_tokens,
                                         chars_per_token, rng)
    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    # ---- turn 1: warm the prefix, response ignored ----------------------
    messages.append({"role": "user",
                     "content": build_user_message(cfg.user_tokens, rng,
                                                    chars_per_token)})
    try:
        await stream_chat(client, cfg.model, messages,
                          max_tokens=cfg.warm_turn1_max_tokens,
                          ignore_eos=False, temperature=cfg.temperature)
    except Exception:  # noqa: BLE001
        pass
    # Append the STATIC turn-1 assistant response (not from the LLM).
    messages.append({"role": "assistant",
                     "content": build_user_message(cfg.static_response_tokens,
                                                    rng, chars_per_token)})

    # ---- turns 2..N: measured ------------------------------------------
    for turn_idx in range(2, cfg.turns + 1):
        gap = rng.uniform(*cfg.inter_turn_gap_s)
        await asyncio.sleep(gap)

        messages.append({"role": "user",
                         "content": build_user_message(cfg.user_tokens, rng,
                                                        chars_per_token)})
        offset = time.perf_counter() - t_step_start
        res = await stream_chat(client, cfg.model, messages,
                                max_tokens=cfg.output_tokens,
                                ignore_eos=cfg.ignore_eos,
                                temperature=cfg.temperature)
        emit(call_idx, turn_idx, offset, res)

        # Carry the model's own output forward as conversation history.
        if res.ok:
            messages.append({"role": "assistant",
                             "content": "(generated) " +
                             build_user_message(cfg.output_tokens, rng,
                                                chars_per_token)})
        else:
            # On failure, still advance history with a placeholder so token
            # accounting stays comparable.
            messages.append({"role": "assistant", "content": "(error)"})
