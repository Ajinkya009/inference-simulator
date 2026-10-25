"""Open-loop Poisson call driver; admission rejection never changes arrivals."""
from __future__ import annotations
import asyncio,random,time
import httpx
from call import run_call
from config import Config
from metrics import StepCollector
from prompt import PromptFactory,make_counter
def _make_client(cfg,transport=None):
 return httpx.AsyncClient(base_url=cfg.base_url,limits=httpx.Limits(max_connections=cfg.max_connections,max_keepalive_connections=cfg.max_connections),timeout=httpx.Timeout(cfg.request_timeout_s,connect=cfg.connect_timeout_s),headers={"Authorization":f"Bearer {cfg.api_key}"},transport=transport)
async def _run_step(cfg,client,lam,factory,rng):
 c=StepCollector(lam); tasks=set(); start=time.monotonic(); measure_start=start+cfg.warmup_s; measure_end=measure_start+cfg.step_duration_s
 def emit(ci,ti,t,res,measured,group,cold):
  # Selection is by request start, not completion; static warm requests stay recorded but unmeasured.
  if measured and measure_start<=t<measure_end:c.add(res,ti,True,call_id=ci,group=group,cold=cold,start_monotonic=t)
 async def wrapped(idx,call_start):
  result=await run_call(idx,client,cfg,factory,random.Random(rng.randrange(2**63)),call_start,emit)
  if measure_start<=call_start<measure_end:c.add_call(result)
 # Poisson offered stream spans warmup plus measurement. It never waits on calls.
 idx=0; next_at=start
 while next_at<measure_end:
  await asyncio.sleep(max(0,next_at-time.monotonic())); offered_at=next_at; in_measure=measure_start<=offered_at<measure_end
  if in_measure:c.offered_calls+=1
  if len(tasks)>=cfg.max_concurrent_calls:
   if in_measure:c.dropped_calls+=1
  else:
   task=asyncio.create_task(wrapped(idx,offered_at));tasks.add(task);task.add_done_callback(tasks.discard);idx+=1
  c.active_samples.append(len(tasks)); next_at+=rng.expovariate(lam)
 # Drain without cancellation; explicitly report unfinished work at deadline.
 deadline=time.monotonic()+cfg.drain_s
 while tasks and time.monotonic()<deadline:
  c.active_samples.append(len(tasks)); await asyncio.sleep(min(.1,max(.001,deadline-time.monotonic())))
 for task in list(tasks): task.cancel()
 await asyncio.gather(*tasks,return_exceptions=True)
 return c.finalize(cfg.step_duration_s,cfg.ttft_slo_ms)
async def run_sweep(cfg:Config,progress=print,transport=None):
 rng=random.Random(cfg.seed);results=[]
 async with _make_client(cfg,transport) as client:
  progress(f"[{cfg.label}] loading tokenizer (tokenizer files only; server fallback if unavailable) ...")
  counter=await make_counter(client,cfg.model,rng,cfg.tokenizer_local_only);factory=PromptFactory(cfg,counter,rng);factory.validate_shared_prefix()
  progress(f"[{cfg.label}] prefix content tokens={counter.count(factory.system_prefix(0)) if cfg.prompt_sharing!='unique' else cfg.shared_prefix_tokens}")
  for lam in cfg.arrival_rates:
   progress(f"[{cfg.label}] offered={lam:.3f} calls/s warmup={cfg.warmup_s:.0f}s measurement={cfg.step_duration_s:.0f}s")
   s=await _run_step(cfg,client,lam,factory,rng);results.append(s)
   p=s.ttft_ms.get("p95");progress(f" -> requests={s.n_turns} err={s.error_rate:.1%} p95={p if p is not None else 'n/a'}ms goodput={s.slo_goodput:.2f}/s drops={s.dropped_calls}")
   if cfg.cooldown_s:await asyncio.sleep(cfg.cooldown_s)
 return results
