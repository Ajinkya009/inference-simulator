# inference-simulator

An asynchronous, open-loop capacity harness for OpenAI-compatible SGLang voice agents. It measures **calls/sec**, not individual LLM turns. Point the same workload at two independent H100 replicas (each at half traffic) and one H200, then compare results.

## Production profile

`voice-production` models 150,000 calls/day: 16,320 shared system-prefix tokens, 680 per-call customer-context tokens, 100 user tokens/turn, 50--100 output tokens, and a 50/50 mix of three/four turns. Default turn offsets are 5/35/70/105 seconds (four turns) and 5/55/105 (three), producing roughly 120-second calls. Later turns never overlap a prior unfinished turn; scheduling slip is retained.

It uses `llm_first_turn`: every turn invokes and measures the LLM. Legacy `static_first_turn` remains for predetermined opening replies. Failure stops a call and is recorded; no invented assistant text is added to history.

`shared` builds each 16,320-token prefix once and reuses it verbatim. Customer context follows the prefix, preserving cross-call caching. `unique` provides cold-prefix stress; `multi-template` chooses stable groups; `partial-miss` creates new prefix variants rather than flushing the server cache. Prefix token IDs are validated. Gemma tokenizer files are used when available, without loading model weights; otherwise the server usage probe calibrates prompt construction.

The complete message history is sent on each turn, including exact assistant text reconstructed from ordered `delta.content` fragments. Empty, role, usage, reasoning/tool, and `[DONE]` chunks do not establish voice TTFT. TTFT means first usable content. Chunk gaps are separately reported: one SSE chunk is never assumed to equal one generated token.

## Run

```bash
python test_selftest.py
```

Production examples (the preset supplies 180 s warmup, 900 s measurement, and 180 s drain):

```bash
# H100: half-traffic average, peak, and 30% spare capacity
python run.py sweep --preset h100 --label H100-avg --rates 0.87 --base-url http://H100:30000/v1
python run.py sweep --preset h100 --label H100-peak --rates 1.74 --base-url http://H100:30000/v1
python run.py sweep --preset h100 --label H100-headroom --rates 2.48 --base-url http://H100:30000/v1

# H200: whole-workload peak and 30% spare capacity
python run.py sweep --preset h200 --label H200-peak --rates 3.47 --base-url http://H200:30000/v1
python run.py sweep --preset h200 --label H200-headroom --rates 4.96 --base-url http://H200:30000/v1
```

Overrides include `--prompt-sharing`, `--shared-prefix-tokens`, `--customer-context-tokens`, `--user-tokens`, `--output-tokens 50:100`, `--three-turn-probability`, `--timing-mode`, `--groups`, `--warmup`, `--measurement`, `--drain`, `--slo`, and `--seed`.

## Reading results

JSON/CSV report offered/admitted/completed calls, offered/completed LLM RPS, errors, admission drops, latency percentiles, per-turn TTFT, cache hit, and generated-token throughput. P95 is over successful measurements; errors/drops stay in the all-attempt SLO-goodput denominator. Missing cache metadata is `null`, never a fictitious 0%.

Capacity is the largest offered rate meeting P95 TTFT <300 ms, minimum sample, error/completion, and admission requirements; compare it to the curvature knee and achieved throughput. Little's Law is a sanity check: 1.736 calls/s × 120 s means about 208 active calls in steady state. Telephone-call concurrency differs from in-flight LLM requests because speech/TTS pauses separate turns.

Synthetic filler and idealized schedules remain limitations; use traces/transcripts when possible. Token-level timings and cache metadata are server-dependent, so unavailable values are not fabricated.
