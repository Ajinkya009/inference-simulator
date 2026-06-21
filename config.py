"""Configuration for the voice-AI load simulator.

The harness is GPU-agnostic: it drives whatever OpenAI-compatible SGLang
endpoint you point it at. To find the knee for L40S vs H100 you run the same
sweep against each endpoint (with a different --label) and compare.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    # ---- endpoint -------------------------------------------------------
    base_url: str = "http://localhost:30000/v1"      # SGLang OpenAI-compat
    model: str = "gemma-4-12b-it"                      # served model name
    api_key: str = "EMPTY"                             # SGLang ignores it
    label: str = "L40S"                                # tag for output files

    # ---- conversation shape (production-faithful) -----------------------
    turns: int = 4                                     # total turns/call
    system_prompt_tokens: int = 18_000                 # unique per CALL
    output_tokens: int = 100                           # generated per turn
    user_tokens: int = 30                              # per user message
    static_response_tokens: int = 40                   # turn-1 static reply
    inter_turn_gap_s: tuple = (5.0, 7.0)               # uniform; user-speak gap

    # Turn 1 is a STATIC response in production. We still send it so SGLang
    # prefills + caches the (unique) 18K prefix, but we cap decode to 1 token
    # and discard it. Only turns 2..N are measured.
    warm_turn1_max_tokens: int = 1

    # Force exactly output_tokens per measured turn so ITL is apples-to-apples
    # across runs (SGLang honours ignore_eos).
    ignore_eos: bool = True
    temperature: float = 0.0

    # ---- load profile (open-loop Poisson arrivals) ----------------------
    # Sweep call-arrival rate lambda (calls/sec). Defaults are seeded from the
    # KV-capacity estimate: ~13-15 warm calls on L40S, ~35-40 on H100, with a
    # call occupying KV for ~turns * (gap + decode) ~= 20-25s (Little's law),
    # so the L40S knee is expected near lambda ~= 0.5-0.7 calls/s.
    arrival_rates: List[float] = field(default_factory=lambda: [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25, 1.5
    ])
    step_duration_s: float = 90.0      # measurement window per lambda
    warmup_s: float = 20.0             # discard samples before this (per step)
    cooldown_s: float = 8.0            # drain in-flight calls between steps
    max_concurrent_calls: int = 2000   # safety cap on open-loop pileup

    # ---- SLO / knee -----------------------------------------------------
    ttft_slo_ms: float = 300.0         # P95 TTFT target (turns 2-4)
    itl_slo_ms: float = 80.0           # P95 ITL target (optional)

    # ---- client ---------------------------------------------------------
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    max_connections: int = 4000

    # ---- output ---------------------------------------------------------
    out_dir: str = "results"
    seed: int = 1234

    def to_dict(self) -> dict:
        d = asdict(self)
        d["inter_turn_gap_s"] = list(self.inter_turn_gap_s)
        return d


# Convenience presets you can load and tweak from the CLI.
PRESETS = {
    "l40s": dict(label="L40S",
                 arrival_rates=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.25]),
    "h100": dict(label="H100",
                 arrival_rates=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]),
    # Fast smoke test against a real or mock endpoint.
    "smoke": dict(arrival_rates=[0.2, 0.5, 1.0],
                  step_duration_s=15.0, warmup_s=3.0, cooldown_s=2.0,
                  system_prompt_tokens=2000),
}
