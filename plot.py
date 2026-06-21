"""Plotting. Optional: if matplotlib is missing we just skip (CSV/JSON always
get written). Supports overlaying multiple result files to compare GPUs."""
from __future__ import annotations

import json
from typing import List, Optional

from .metrics import StepStats, slo_knee, kneedle_knee


def _load_series(rows: List[dict]):
    lam = [r["arrival_rate"] for r in rows]
    p95 = [r["ttft_p95"] for r in rows]
    p50 = [r["ttft_p50"] for r in rows]
    itl95 = [r["itl_p95"] for r in rows]
    return lam, p50, p95, itl95


def plot_single(steps: List[StepStats], rows: List[dict], cfg, path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping plot")
        return None

    lam, p50, p95, itl95 = _load_series(rows)
    sk = slo_knee(steps, cfg.ttft_slo_ms)
    kk = kneedle_knee(steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(lam, p95, "o-", color="#d62728", label="TTFT P95")
    ax1.plot(lam, p50, "o--", color="#ff9896", label="TTFT P50", alpha=0.8)
    ax1.axhline(cfg.ttft_slo_ms, color="gray", ls=":", label=f"SLO {cfg.ttft_slo_ms:.0f}ms")
    if sk is not None:
        ax1.axvline(sk, color="green", ls="-", alpha=0.6, label=f"SLO knee λ={sk:.2f}")
    if kk is not None:
        ax1.axvline(kk, color="purple", ls="--", alpha=0.6, label=f"curvature knee λ={kk:.2f}")
    ax1.set_xlabel("arrival rate λ (calls/s)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title(f"{cfg.label}: TTFT P95 vs load (turns 2-{cfg.turns})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(lam, itl95, "s-", color="#1f77b4", label="ITL P95")
    ax2.axhline(cfg.itl_slo_ms, color="gray", ls=":", label=f"ITL SLO {cfg.itl_slo_ms:.0f}ms")
    ax2.set_xlabel("arrival rate λ (calls/s)")
    ax2.set_ylabel("ITL (ms)")
    ax2.set_title(f"{cfg.label}: inter-token latency P95")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[plot] wrote {path}")
    return path


def plot_compare(result_files: List[str], out_path: str,
                 slo_ms: float = 300.0):
    """Overlay TTFT P95 vs λ across multiple result JSON files (e.g. L40S vs H100)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping compare plot")
        return None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for i, f in enumerate(result_files):
        with open(f) as fh:
            blob = json.load(fh)
        rows = blob["rows"]
        label = blob.get("config", {}).get("label", f)
        lam = [r["arrival_rate"] for r in rows]
        p95 = [r["ttft_p95"] for r in rows]
        c = colors[i % len(colors)]
        ax.plot(lam, p95, "o-", color=c, label=f"{label} TTFT P95")
        knee = blob.get("slo_knee")
        if knee:
            ax.axvline(knee, color=c, ls="--", alpha=0.5)
    ax.axhline(slo_ms, color="gray", ls=":", label=f"SLO {slo_ms:.0f}ms")
    ax.set_xlabel("arrival rate λ (calls/s)")
    ax.set_ylabel("TTFT P95 (ms)")
    ax.set_title("Knee comparison")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"[plot] wrote {out_path}")
    return out_path
