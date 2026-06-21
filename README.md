# inference-simulator — LLM serving load simulator (knee finder)

A production-faithful, open-loop load generator for a self-hosted **Gemma 4 12B-it**
voice agent on SGLang. It drives a real OpenAI-compatible endpoint with Poisson
call arrivals, measures **P95 TTFT and ITL on turns 2–N**, and locates the
**knee** of the TTFT-vs-load curve for whatever GPU is behind the endpoint.

It's GPU-agnostic: run the same sweep against an L40S endpoint and an H100
endpoint, then overlay the two curves to compare knees.

## What it models (and why)

Each Poisson arrival is one **call** with a **unique 18K-token system prompt**
(unique per call, shared across that call's turns — never across calls). Per call:

- **Turn 1** sends `[system(18K) + user]`, decodes 1 token, and **discards it**.
  In production the turn-1 reply is static; here it exists only to **prefill and
  warm the call's unique prefix** in SGLang's RadixAttention cache.
- **5–7 s gap** (user speaking / ASR) between turns — this is when a call's KV
  sits idle and becomes eligible for LRU eviction.
- **Turns 2..N** carry **full history forward** and generate **100 tokens** each.
  These are the only **measured** turns. Turn 2+ should hit the warm prefix
  (cache hit), so a clean knee reflects when **KV capacity + compute** stop
  keeping up and eviction forces cold re-prefills.

Why the knee is where it is on an L40S: Gemma 4 12B uses **hybrid attention** —
most of its 48 layers are local sliding-window (window 1024), so at 18K context
only the few global layers cache the full prompt. That puts per-call KV at
roughly **1.3–1.5 GB**, so a 48 GB L40S (~24 GB weights) holds only **~13–15
warm calls** before eviction churns. An 80 GB H100 holds ~35–40 plus faster
prefill/decode, so its knee lands much higher. The harness measures the truth;
those numbers only seed sensible sweep ranges.

## Assumptions (all configurable in `config.py` or via CLI)

| Knob | Default | Notes |
|---|---|---|
| system prompt | 18,000 tok, unique/call | UUID-prefixed → no cross-call sharing |
| turns | 4 (turn 1 warm, 2–4 measured) | |
| output tokens | 100, `ignore_eos=True` | forces exact length for clean ITL |
| inter-turn gap | uniform 5–7 s | |
| arrivals | open-loop Poisson, sweep λ | exponential inter-arrival |
| SLO | P95 TTFT ≤ 300 ms | knee #1 = largest λ meeting this |

If `ignore_eos` isn't desired (you want natural EOS), set `--rates` aside and
flip it in config; ITL stats then vary with real output length.

## Install

```bash
uv venv && source .venv/bin/activate     # or your existing env
pip install -r requirements.txt          # httpx, numpy, matplotlib(optional)
```

## Use

**1. Validate plumbing first** (one call; confirms the warm prefix actually
hits the cache before you spend GPU-hours):

```bash
python run.py validate --base-url http://YOUR_RUNPOD:30000/v1 --model gemma-4-12b-it
```
Look for `turn2 ... cached=~18000` and the `OK: turn-2 reused the warm prefix`
line. If cache hit is low, prefix caching isn't engaging — fix that first.

**2. Sweep to find the knee:**

```bash
python run.py sweep --base-url http://YOUR_RUNPOD:30000/v1 \
    --model gemma-4-12b-it --preset l40s --label L40S
# then on the H100 box:
python run.py sweep --base-url http://YOUR_H100:30000/v1 \
    --model gemma-4-12b-it --preset h100 --label H100
```
Writes `results/<LABEL>.{json,csv,png}` and prints a knee summary.

**3. Compare GPUs:**

```bash
python run.py compare results/L40S.json results/H100.json -o results/compare.png
```

Handy flags: `--rates 0.2,0.5,0.8,1.0,1.5` (custom λ sweep),
`--system-prompt-tokens 18000`, `--step-duration 90`, `--slo 300`.

## Reading the output

Per λ step the CSV/JSON has: `ttft_p50/p95/p99`, `itl_p50/p95/p99`,
`throughput_turns_s`, `output_tok_s`, `cache_hit_rate`, `achieved_concurrency`
(Little's-law cross-check: λ_turn × mean turn time), `error_rate`.

Two knees are reported:
- **SLO knee** — largest λ with P95 TTFT ≤ SLO and errors < 2%. The number you
  provision against.
- **Curvature knee** — Kneedle-style elbow of the P95 curve, even if it never
  crosses the SLO. Catches the inflection where things start degrading.

Watch `cache_hit_rate`: when it collapses as λ rises, that's eviction kicking in
— usually the same λ where TTFT P95 bends. That correlation is the mechanism.

## Self-test (no GPU)

`python test_selftest.py` runs the full pipeline against an in-process streaming
mock with load-dependent TTFT, asserting that streaming timing capture, the
Poisson driver, cache-hit parsing, and knee detection all work.
`mock_server.py` is a standalone network version of the same mock if you want to
hit it over HTTP.

## Caveats

- Open-loop means that past the knee, calls pile up and latency grows
  unbounded — that's intended (it reveals the knee). `max_concurrent_calls`
  caps client-side pileup so the *generator* doesn't OOM during collapse.
- Filler text approximates real Hindi prompt token distribution; token *counts*
  are calibrated against the server, but token *content* is synthetic. If you
  have real CDR transcripts, swap `prompt.py`'s filler for sampled turns.
- `cached_tokens` parsing assumes SGLang's OpenAI-compatible
  `usage.prompt_tokens_details.cached_tokens`. If your build omits it,
  cache-hit rate reads 0 but TTFT/ITL/knee are unaffected.
