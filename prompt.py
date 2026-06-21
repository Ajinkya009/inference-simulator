"""Prompt construction.

Requirements from the production profile:

1. The system prompt is UNIQUE per call (not shared across calls). We guarantee
   no accidental cross-call RadixAttention prefix sharing by making the very
   first tokens a per-call UUID. Within a call the identical system prompt is
   resent every turn, so the per-call prefix IS reused across turns 2..N.

2. We must land near the target token count without the gated Gemma-4 tokenizer.
   We calibrate ONCE against the server's usage.prompt_tokens to get an accurate
   tokens-per-word ratio, then size prompts in WORDS. (Earlier versions routed
   through chars/token with a hardcoded chars/word and overshot ~40%.)
"""
from __future__ import annotations

import random
import uuid
from typing import List

_WORDS = (
    "system policy customer account balance overdue payment reminder schedule "
    "agent collections loan installment principal interest tenure escalation "
    "verify identity consent disclosure compliance regulatory script empathy "
    "negotiate settlement promise commitment callback followup resolution note "
    "context history preference language Hindi Marathi Tamil Kannada Spanish "
    "branch region tier priority sentiment objection rebuttal clarify confirm"
).split()

_PROBE_WORDS = 4000          # calibration probe size
_HEADER_TOKENS = 45          # rough token cost of the system header below
_DEFAULT_TPW = 1.0           # fallback tokens-per-word if the probe fails


def _filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def build_system_prompt(target_tokens: int, tpw: float,
                        rng: random.Random) -> str:
    """Unique system prompt of ~target_tokens. `tpw` = tokens per word."""
    call_id = uuid.uuid4().hex  # unique prefix -> no cross-call sharing
    header = (
        f"### CALL {call_id}\n"
        "You are a Hindi-language voice collections agent. Follow policy "
        "strictly and keep responses concise.\n### KNOWLEDGE BASE\n"
    )
    n_words = max(1, round((target_tokens - _HEADER_TOKENS) / max(tpw, 1e-6)))
    return header + _filler(n_words, rng)


def build_user_message(n_tokens: int, tpw: float,
                       rng: random.Random) -> str:
    n_words = max(1, round(n_tokens / max(tpw, 1e-6)))
    return _filler(n_words, rng)


async def calibrate(client, model: str, rng: random.Random) -> float:
    """Return tokens-per-word measured on the live server.

    Sends one cheap probe (max_tokens=1) of known word count and reads
    usage.prompt_tokens. Falls back to 1.0 if usage is unavailable.
    """
    probe_text = _filler(_PROBE_WORDS, rng)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": probe_text}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
        "return_cached_tokens_details": True,
    }
    try:
        r = await client.post("/chat/completions", json=payload)
        r.raise_for_status()
        usage = r.json().get("usage") or {}
        pt = usage.get("prompt_tokens")
        if pt:
            # prompt_tokens includes a few template tokens; negligible at 4k words
            return pt / _PROBE_WORDS
    except Exception as e:  # noqa: BLE001
        print(f"[calibrate] probe failed ({e}); using default {_DEFAULT_TPW} tok/word")
    return _DEFAULT_TPW


def measured_prompt_tokens(usage: dict) -> int:
    return int((usage or {}).get("prompt_tokens", 0))


def cached_prompt_tokens(usage: dict) -> int:
    """SGLang reports prefix-cache hits via prompt_tokens_details.cached_tokens
    (requires return_cached_tokens_details=True on the request)."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    for k in ("cached_tokens", "cached"):
        if k in details:
            return int(details[k])
    return int(usage.get("cached_tokens", 0))
