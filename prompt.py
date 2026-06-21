"""Prompt construction.

Two hard requirements from the production profile:

1. The 18K system prompt is UNIQUE per call (not shared across calls). We
   guarantee no accidental cross-call RadixAttention prefix sharing by making
   the very first tokens a per-call UUID. Within a call the identical system
   prompt is resent every turn, so the per-call prefix *is* reused across
   turns 2..N (warm path) -- which is the whole point.

2. We must land near a real 18K token count without the gated Gemma-4
   tokenizer. We calibrate once at startup against the server's reported
   usage.prompt_tokens and cache a chars-per-token ratio.
"""
from __future__ import annotations

import random
import uuid
from typing import List

# A varied word bank so filler tokenizes at a realistic (sub-1.0) ratio rather
# than collapsing into repeated single tokens.
_WORDS = (
    "system policy customer account balance overdue payment reminder schedule "
    "agent collections loan installment principal interest tenure escalation "
    "verify identity consent disclosure compliance regulatory script empathy "
    "negotiate settlement promise commitment callback followup resolution note "
    "context history preference language Hindi Marathi Tamil Kannada Spanish "
    "branch region tier priority sentiment objection rebuttal clarify confirm"
).split()


def _filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def build_system_prompt(target_tokens: int, chars_per_token: float,
                        rng: random.Random) -> str:
    """Build a unique system prompt of ~target_tokens.

    chars_per_token comes from calibrate(); ~4.0 is a safe initial guess.
    """
    call_id = uuid.uuid4().hex  # unique prefix -> no cross-call sharing
    header = (
        f"### CALL {call_id}\n"
        "You are a Hindi-language voice collections agent. Follow policy "
        "strictly and keep responses concise.\n### KNOWLEDGE BASE\n"
    )
    remaining_chars = max(0, int(target_tokens * chars_per_token) - len(header))
    # ~6 chars per word incl. space.
    n_words = max(1, remaining_chars // 6)
    return header + _filler(n_words, rng)


def build_user_message(n_tokens: int, rng: random.Random,
                       chars_per_token: float) -> str:
    n_words = max(1, int(n_tokens * chars_per_token) // 6)
    return _filler(n_words, rng)


async def calibrate(client, model: str, rng: random.Random) -> float:
    """Measure chars-per-token on the live server. Returns chars/token.

    Sends one cheap probe (max_tokens=1) and reads usage.prompt_tokens.
    Falls back to 4.0 if the server doesn't return usage.
    """
    probe_text = _filler(4000, rng)  # ~4k words
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": probe_text}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        r = await client.post("/chat/completions", json=payload)
        r.raise_for_status()
        usage = r.json().get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens:
            return len(probe_text) / prompt_tokens
    except Exception as e:  # noqa: BLE001
        print(f"[calibrate] probe failed ({e}); using default 4.0 chars/token")
    return 4.0


def measured_prompt_tokens(usage: dict) -> int:
    return int((usage or {}).get("prompt_tokens", 0))


def cached_prompt_tokens(usage: dict) -> int:
    """SGLang reports prefix-cache hits via prompt_tokens_details.cached_tokens
    (OpenAI-compatible). Defensive parse across field variants."""
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    for k in ("cached_tokens", "cached"):
        if k in details:
            return int(details[k])
    return int(usage.get("cached_tokens", 0))
