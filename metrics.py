"""Aggregation with all-attempt denominators for SLO goodput."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from typing import List,Optional
import numpy as np
from client import TurnResult
def _pct(xs):
    if not xs:return {k:None for k in ("p50","p90","p95","p99","mean","max")}
    a=np.asarray(xs,float);return {"p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"mean":float(a.mean()),"max":float(a.max())}
@dataclass
class StepStats:
    arrival_rate:float; n_turns:int=0; n_errors:int=0; offered_calls:int=0; admitted_calls:int=0; completed_calls:int=0; dropped_calls:int=0; incomplete_calls:int=0
    ttft_ms:dict=field(default_factory=dict); e2e_ms:dict=field(default_factory=dict); itl_ms:dict=field(default_factory=dict); per_turn_ttft:dict=field(default_factory=dict)
    throughput_turns_s:float=0.; offered_calls_s:float=0.; admitted_calls_s:float=0.; completed_calls_s:float=0.; offered_llm_rps:float=0.; completed_llm_rps:float=0.; output_tok_s:float=0.
    cache_hit_rate:Optional[float]=None; mean_prompt_tokens:float=0.; achieved_concurrency:float=0.; error_rate:float=0.; slo_goodput:float=0.; mean_active_calls:float=0.; peak_active_calls:int=0; mean_inflight_requests:float=0.; peak_inflight_requests:int=0; call_duration_s:dict=field(default_factory=dict)
    request_records:List[dict]=field(default_factory=list)
class StepCollector:
 def __init__(self,rate):
    self.arrival_rate=rate;self.records=[];self.calls=[];self.offered_calls=0;self.dropped_calls=0;self.active_samples=[];self.inflight_samples=[]
 def add(self,res,turn=0,measured=True,**extra): self.records.append((res,turn,measured,extra))
 def add_call(self,c):self.calls.append(c)
 def finalize(self,window_s,slo_ms=300):
    measured=[x for x in self.records if x[2]]; ok=[x for x in measured if x[0].ok]; rs=[x[0] for x in ok]
    pts=[r.prompt_tokens for r in rs if r.prompt_tokens]; caches=[r.cached_tokens for r in rs if r.cached_tokens is not None]
    by={str(t):_pct([r.ttft_ms for r,tt,_,_ in ok if tt==t]) for t in range(1,5)}
    completed=sum(1 for c in self.calls if c.get("completed")); durations=[c["duration_s"] for c in self.calls]
    n=len(measured); good=sum(1 for r in rs if r.ttft_ms<slo_ms)
    records=[{"turn":t,"measured":m,"ok":r.ok,"ttft_ms":r.ttft_ms if r.ok else None,"e2e_ms":r.total_s*1000,"prompt_tokens":r.prompt_tokens,"cached_tokens":r.cached_tokens,"fresh_tokens":r.fresh_tokens,"completion_tokens":r.completion_tokens,"chunk_gaps_ms":r.chunk_gaps_ms,"error":r.error,**e} for r,t,m,e in self.records]
    return StepStats(self.arrival_rate,n,len(measured)-len(ok),self.offered_calls,len(self.calls),completed,self.dropped_calls,max(0,len(self.calls)-completed),_pct([r.ttft_ms for r in rs]),_pct([r.total_s*1000 for r in rs]),_pct([g for r in rs for g in r.chunk_gaps_ms]),by,len(ok)/window_s,self.offered_calls/window_s,len(self.calls)/window_s,completed/window_s,n/window_s,len(ok)/window_s,sum((r.completion_tokens or 0) for r in rs)/window_s,(sum(caches)/sum(pts) if pts and len(caches)==len(rs) else None),(sum(pts)/len(pts) if pts else 0.),0.,((n-len(ok))/n if n else 0.),good/window_s,(sum(self.active_samples)/len(self.active_samples) if self.active_samples else 0.),max(self.active_samples or [0]),(sum(self.inflight_samples)/len(self.inflight_samples) if self.inflight_samples else 0.),max(self.inflight_samples or [0]),_pct(durations),records)
def slo_knee(steps,slo_ms,min_samples=20,max_error_rate=.02):
 p=[s.arrival_rate for s in steps if s.n_turns>=min_samples and s.ttft_ms.get("p95") is not None and s.ttft_ms["p95"]<=slo_ms and s.error_rate<=max_error_rate and not s.dropped_calls]
 return max(p) if p else None
def kneedle_knee(steps):
 pts=[(s.arrival_rate,s.ttft_ms.get("p95")) for s in steps if s.ttft_ms.get("p95") is not None]
 if len(pts)<3:return None
 x=np.array([a for a,b in pts]);y=np.array([b for a,b in pts]);xn=(x-x.min())/(np.ptp(x)or 1);yn=(y-y.min())/(np.ptp(y)or 1);d=np.abs((yn[-1]-yn[0])*xn-(xn[-1]-xn[0])*yn+xn[-1]*yn[0]-yn[-1]*xn[0])/ (np.hypot(yn[-1]-yn[0],xn[-1]-xn[0])or 1);return float(x[d.argmax()])
def steps_to_rows(steps):
 out=[]
 for s in steps:
  d=asdict(s)
  d.pop("request_records", None)
  for n in ("ttft","e2e","itl"):
   for k,v in d.pop(n+"_ms").items():d[f"{n}_{k}"]=v
  out.append(d)
 return out
