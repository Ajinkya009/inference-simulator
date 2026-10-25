"""One telephone call and its sequential multi-turn LLM lifecycle."""
from __future__ import annotations
import asyncio, random, time
from typing import Callable
from client import stream_chat, TurnResult

async def run_call(call_idx, client, cfg, factory, rng, call_start, emit:Callable):
    """Run one call. `emit` receives every attempted LLM turn, including errors.

    Failure policy is stop-call: a failed response is recorded and no invented
    assistant message is put into subsequent history.
    """
    nturns=3 if rng.random()<cfg.three_turn_probability else cfg.turns
    group=factory.group_for(rng); cold=(cfg.prompt_sharing=="partial-miss" and rng.random()<cfg.cold_miss_percent/100.)
    messages=[{"role":"system","content":factory.system_prefix(group,cold)}]
    context=factory.customer_context(rng)
    if context: messages.append({"role":"system","content":context})
    schedule=cfg.turn_schedule_3 if nturns==3 else cfg.turn_schedule_4
    completed=True
    for turn in range(1,nturns+1):
        if cfg.timing_mode=="scheduled":
            target=call_start+schedule[min(turn-1,len(schedule)-1)]+rng.uniform(-cfg.schedule_jitter_s,cfg.schedule_jitter_s)
            await asyncio.sleep(max(0.,target-time.monotonic()))
        elif cfg.timing_mode=="fixed-gap" and turn>1: await asyncio.sleep(rng.uniform(*cfg.inter_turn_gap_s))
        if cfg.customer_context_each_turn and turn>1 and context: messages.append({"role":"system","content":factory.customer_context(rng)})
        messages.append({"role":"user","content":factory.user(rng)})
        start=time.monotonic()
        if turn==1 and cfg.first_turn_mode=="static_first_turn":
            # Legacy behavior remains explicit: prefill request is recorded but not a measured LLM reply.
            res=await stream_chat(client,cfg.model,messages,cfg.warm_turn1_max_tokens,False,cfg.temperature)
            emit(call_idx,turn,start,res,False,group,cold)
            if not res.ok: completed=False; break
            messages.append({"role":"assistant","content":factory.user(rng)})
            continue
        maximum=rng.randint(cfg.output_tokens_min,cfg.output_tokens_max)
        res=await stream_chat(client,cfg.model,messages,maximum,cfg.ignore_eos,cfg.temperature)
        emit(call_idx,turn,start,res,True,group,cold)
        if not res.ok:
            completed=False; break
        # Exact reconstructed response is the next-turn assistant context.
        messages.append({"role":"assistant","content":res.assistant_text})
    return {"call_id":call_idx,"turns":nturns,"completed":completed,"duration_s":time.monotonic()-call_start,"group":group,"cold":cold}
