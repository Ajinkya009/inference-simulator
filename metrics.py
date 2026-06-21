"""Metric aggregation and knee detection.

For each lambda step we collect TTFT and ITL samples from measured turns (2..N,
post-warmup) and compute P50/P95/P99, throughput, error rate, cache-hit rate,
and achieved concurrency (Little's law cross-check).

Two knee definitions, both reported:
  * SLO knee   : largest lambda whose P95 TTFT <= ttft_slo_ms. Most actionable.
  * Kneedle knee: max-curvature elbow of the P95-TTFT-vs-lambda curve, even if
                  it never crosses the SLO. Catches the inflection.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from .client import TurnResult


@dataclass
class StepStats:
    arrival_rate: float
    n_turns: int = 0
    n_errors: int = 0
    ttft_ms: dict = field(default_factory=dict)
    itl_ms: dict = field(default_factory=dict)
    tpot_ms_mean: float = 0.0
    throughput_turns_s: float = 0.0
    output_tok_s: float = 0.0
    cache_hit_rate: float = 0.0
    mean_prompt_tokens: float = 0.0
    achieved_concurrency: float = 0.0   # Little's law: lambda_turn * mean_total_s
    error_rate: float = 0.0


def _pct(xs: List[float]) -> dict:
    if not xs:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    a = np.asarray(xs, dtype=float)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "max": float(a.max()),
    }


class StepCollector:
    """Accumulates samples for one lambda step (warmup already filtered)."""

    def __init__(self, arrival_rate: float):
        self.arrival_rate = arrival_rate
        self.ttft: List[float] = []
        self.itl: List[float] = []
        self.tpot: List[float] = []
        self.total_s: List[float] = []
        self.prompt_tokens: List[int] = []
        self.cached_tokens: List[int] = []
        self.n_turns = 0
        self.n_errors = 0
        self.n_output_tokens = 0

    def add(self, res: TurnResult):
        self.n_turns += 1
        if not res.ok:
            self.n_errors += 1
            return
        self.ttft.append(res.ttft_ms)
        self.itl.extend(res.itls_ms)
        if res.tpot_ms:
            self.tpot.append(res.tpot_ms)
        self.total_s.append(res.total_s)
        self.prompt_tokens.append(res.prompt_tokens)
        self.cached_tokens.append(res.cached_tokens)
        self.n_output_tokens += res.n_output

    def finalize(self, window_s: float) -> StepStats:
        ok = self.n_turns - self.n_errors
        pt = sum(self.prompt_tokens)
        ct = sum(self.cached_tokens)
        mean_total = statistics.mean(self.total_s) if self.total_s else 0.0
        turn_rate = ok / window_s if window_s > 0 else 0.0
        return StepStats(
            arrival_rate=self.arrival_rate,
            n_turns=self.n_turns,
            n_errors=self.n_errors,
            ttft_ms=_pct(self.ttft),
            itl_ms=_pct(self.itl),
            tpot_ms_mean=statistics.mean(self.tpot) if self.tpot else 0.0,
            throughput_turns_s=turn_rate,
            output_tok_s=self.n_output_tokens / window_s if window_s > 0 else 0.0,
            cache_hit_rate=(ct / pt) if pt else 0.0,
            mean_prompt_tokens=statistics.mean(self.prompt_tokens) if self.prompt_tokens else 0.0,
            achieved_concurrency=turn_rate * mean_total,
            error_rate=(self.n_errors / self.n_turns) if self.n_turns else 0.0,
        )


def slo_knee(steps: List[StepStats], slo_ms: float) -> Optional[float]:
    """Largest arrival rate with P95 TTFT <= slo and no error blowup."""
    passing = [s.arrival_rate for s in steps
               if s.ttft_ms.get("p95", 1e9) <= slo_ms and s.error_rate < 0.02]
    return max(passing) if passing else None


def kneedle_knee(steps: List[StepStats]) -> Optional[float]:
    """Max-curvature elbow of P95 TTFT vs arrival rate (normalized chord dist)."""
    pts = [(s.arrival_rate, s.ttft_ms.get("p95", 0.0)) for s in steps]
    pts = [p for p in pts if p[1] > 0]
    if len(pts) < 3:
        return None
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    xn = (xs - xs.min()) / (np.ptp(xs) or 1)
    yn = (ys - ys.min()) / (np.ptp(ys) or 1)
    # perpendicular distance of each point from the chord (first->last)
    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    num = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0) or 1
    dist = num / den
    return float(xs[int(dist.argmax())])


def steps_to_rows(steps: List[StepStats]) -> List[dict]:
    rows = []
    for s in steps:
        d = asdict(s)
        d["ttft_p50"] = s.ttft_ms["p50"]
        d["ttft_p95"] = s.ttft_ms["p95"]
        d["ttft_p99"] = s.ttft_ms["p99"]
        d["itl_p50"] = s.itl_ms["p50"]
        d["itl_p95"] = s.itl_ms["p95"]
        d["itl_p99"] = s.itl_ms["p99"]
        del d["ttft_ms"], d["itl_ms"]
        rows.append(d)
    return rows
