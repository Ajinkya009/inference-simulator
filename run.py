"""CLI.

  python -m inference_simulator sweep    --base-url http://HOST:30000/v1 --preset l40s --label L40S
  python -m inference_simulator validate --base-url http://HOST:30000/v1   # 1 call, prints breakdown
  python -m inference_simulator compare results/L40S.json results/H100.json -o results/compare.png
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time

from config import Config, PRESETS
from runner import run_sweep, _make_client
from metrics import slo_knee, kneedle_knee, steps_to_rows
from prompt import calibrate, build_system_prompt, build_user_message
from call import run_call
from client import stream_chat


def _apply_cli(cfg: Config, args) -> Config:
    if getattr(args, "preset", None):
        for k, v in PRESETS[args.preset].items():
            setattr(cfg, k, v)
    for attr in ("base_url", "model", "label", "out_dir"):
        v = getattr(args, attr, None)
        if v is not None:
            setattr(cfg, attr, v)
    if getattr(args, "system_prompt_tokens", None):
        cfg.system_prompt_tokens = args.system_prompt_tokens
    if getattr(args, "rates", None):
        cfg.arrival_rates = [float(x) for x in args.rates.split(",")]
    if getattr(args, "step_duration", None):
        cfg.step_duration_s = args.step_duration
    if getattr(args, "slo", None):
        cfg.ttft_slo_ms = args.slo
    return cfg


def _write_outputs(cfg: Config, steps):
    os.makedirs(cfg.out_dir, exist_ok=True)
    rows = steps_to_rows(steps)
    sk = slo_knee(steps, cfg.ttft_slo_ms)
    kk = kneedle_knee(steps)
    base = os.path.join(cfg.out_dir, cfg.label)

    blob = {"config": cfg.to_dict(), "rows": rows,
            "slo_knee": sk, "curvature_knee": kk,
            "timestamp": time.time()}
    with open(base + ".json", "w") as f:
        json.dump(blob, f, indent=2)

    # CSV
    import csv
    if rows:
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    try:
        from plot import plot_single
        plot_single(steps, rows, cfg, base + ".png")
    except ImportError:
        print("[plot] plot.py not present; skipping PNG (CSV/JSON written)")

    print("\n" + "=" * 64)
    print(f"  {cfg.label} KNEE SUMMARY")
    print("=" * 64)
    print(f"  SLO knee (P95 TTFT <= {cfg.ttft_slo_ms:.0f}ms): "
          f"{('λ=%.3f calls/s' % sk) if sk else 'NONE (SLO never met)'}")
    print(f"  Curvature knee (elbow):           "
          f"{('λ=%.3f calls/s' % kk) if kk else 'n/a'}")
    print(f"  Outputs: {base}.json / .csv / .png")
    print("=" * 64)


async def _validate(cfg: Config):
    rng = random.Random(cfg.seed)
    async with _make_client(cfg) as client:
        print(f"[validate] calibrating against {cfg.base_url} ...")
        cpt = await calibrate(client, cfg.model, rng)
        sp = build_system_prompt(cfg.system_prompt_tokens, cpt, rng)
        print(f"[validate] tokens/word={cpt:.3f}  sys_prompt_chars={len(sp)}")

        # one warm + measured turn to confirm plumbing & cache behavior
        msgs = [{"role": "system", "content": sp},
                {"role": "user", "content": build_user_message(cfg.user_tokens, cpt, rng)}]
        warm = await stream_chat(client, cfg.model, msgs, cfg.warm_turn1_max_tokens,
                                 False, cfg.temperature)
        print(f"[validate] turn1 (warm): ok={warm.ok} prompt_tok={warm.prompt_tokens} "
              f"cached={warm.cached_tokens} err={warm.error}")
        msgs.append({"role": "assistant", "content": build_user_message(40, cpt, rng)})
        msgs.append({"role": "user", "content": build_user_message(cfg.user_tokens, cpt, rng)})
        m = await stream_chat(client, cfg.model, msgs, cfg.output_tokens,
                              cfg.ignore_eos, cfg.temperature)
        print(f"[validate] turn2 (measured): ok={m.ok} TTFT={m.ttft_ms:.0f}ms "
              f"ITL_p95={sorted(m.itls_ms)[int(len(m.itls_ms)*0.95)] if m.itls_ms else 0:.1f}ms "
              f"n_out={m.n_output} prompt_tok={m.prompt_tokens} cached={m.cached_tokens}")
        if m.ok and m.cached_tokens >= cfg.system_prompt_tokens * 0.8:
            print("[validate] OK: turn-2 reused the warm prefix (cache hit). "
                  "Knee sweep will be meaningful.")
        elif m.ok:
            print("[validate] WARN: low cache hit on turn 2 -- check that prefix "
                  "caching/RadixAttention is enabled on the server.")


def main():
    p = argparse.ArgumentParser(prog="inference-simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--base-url", default=None)
        sp.add_argument("--model", default=None)
        sp.add_argument("--label", default=None)
        sp.add_argument("--preset", choices=list(PRESETS.keys()))
        sp.add_argument("--system-prompt-tokens", type=int)
        sp.add_argument("--rates", help="comma list, e.g. 0.2,0.5,1.0")
        sp.add_argument("--step-duration", type=float)
        sp.add_argument("--slo", type=float, help="P95 TTFT SLO in ms")
        sp.add_argument("--out-dir", default=None)

    sw = sub.add_parser("sweep"); common(sw)
    va = sub.add_parser("validate"); common(va)
    cm = sub.add_parser("compare")
    cm.add_argument("files", nargs="+")
    cm.add_argument("-o", "--out", default="results/compare.png")
    cm.add_argument("--slo", type=float, default=300.0)

    args = p.parse_args()

    if args.cmd == "compare":
        from plot import plot_compare
        plot_compare(args.files, args.out, args.slo)
        return

    cfg = _apply_cli(Config(), args)
    if args.cmd == "validate":
        asyncio.run(_validate(cfg))
    elif args.cmd == "sweep":
        steps = asyncio.run(run_sweep(cfg))
        _write_outputs(cfg, steps)


if __name__ == "__main__":
    main()
