"""Token-calibrated, cache-safe prompt construction."""
from __future__ import annotations
import random, uuid
from dataclasses import dataclass
from typing import Dict, List, Optional
_WORDS=("policy account payment reminder schedule agent collections loan installment principal interest compliance disclosure empathy settlement callback history language customer service verify consent context response clarify confirm").split()
def _filler(n,rng): return " ".join(rng.choice(_WORDS) for _ in range(max(1,n)))
@dataclass
class TokenCounter:
    tpw: float=1.; tokenizer: object|None=None
    def count(self,text): return len(self.tokenizer.encode(text,add_special_tokens=False)) if self.tokenizer else max(1,round(len(text.split())*self.tpw))
    def ids(self,text): return list(self.tokenizer.encode(text,add_special_tokens=False)) if self.tokenizer else [hash(w)&0x7fffffff for w in text.split()]
def _to_target(target,counter,rng,header=""):
    text=header+_filler(max(1,round(target/max(counter.tpw,.01))),rng)
    for _ in range(8):
        delta=target-counter.count(text)
        if abs(delta)<=1: break
        words=text.split()
        if delta>0: words.extend(rng.choice(_WORDS) for _ in range(max(1,round(delta/max(counter.tpw,.01)))))
        else: words=words[:max(1,len(words)-max(1,round(-delta/max(counter.tpw,.01))))]
        text=" ".join(words)
    return text
class PromptFactory:
    def __init__(self,cfg,counter,rng):
        self.cfg,self.counter,self.rng=cfg,counter,rng; self.prefixes:Dict[int,str]={}
        if cfg.prompt_sharing!="unique":
            for group in range(cfg.prompt_groups): self.prefixes[group]=_to_target(cfg.shared_prefix_tokens,counter,random.Random(cfg.seed+group),"You are a voice AI agent. Follow policy.\n")
    def group_for(self,rng): return rng.choices(range(self.cfg.prompt_groups),weights=self.cfg.prompt_group_proportions or [1.]*self.cfg.prompt_groups,k=1)[0]
    def system_prefix(self,group,cold=False):
        if self.cfg.prompt_sharing=="unique" or cold: return _to_target(self.cfg.shared_prefix_tokens,self.counter,random.Random(uuid.uuid4().int),"CALL "+uuid.uuid4().hex+" policy voice agent\n")
        return self.prefixes[group]
    def customer_context(self,rng): return _to_target(self.cfg.customer_context_tokens,self.counter,rng,"Customer context: ") if self.cfg.customer_context_tokens else ""
    def user(self,rng): return _to_target(self.cfg.user_tokens,self.counter,rng)
    def validate_shared_prefix(self):
        if self.cfg.prompt_sharing!="unique":
            for g,p in self.prefixes.items():
                if self.counter.ids(p)!=self.counter.ids(self.system_prefix(g)): raise RuntimeError("shared prefix token IDs are not stable")
                if abs(self.counter.count(p)-self.cfg.shared_prefix_tokens)>self.cfg.token_tolerance: raise RuntimeError("shared prefix outside configured token tolerance")
async def calibrate(client,model,rng):
    text=_filler(4000,rng)
    try:
        r=await client.post("/chat/completions",json={"model":model,"messages":[{"role":"user","content":text}],"max_tokens":1,"stream":False}); r.raise_for_status(); return (r.json().get("usage") or {}).get("prompt_tokens",4000)/4000
    except Exception:return 1.
async def make_counter(client,model,rng,local_only=False):
    try:
        from transformers import AutoTokenizer
        return TokenCounter(tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=local_only))
    except Exception:return TokenCounter(tpw=await calibrate(client,model,rng))
def measured_prompt_tokens(usage): return int((usage or {}).get("prompt_tokens") or 0)
def cached_prompt_tokens(usage)->Optional[int]:
    d=(usage or {}).get("prompt_tokens_details") or {}
    for k in ("cached_tokens","cached"):
        if k in d:return int(d[k])
    return int(usage["cached_tokens"]) if usage and "cached_tokens" in usage else None
# legacy API
def build_system_prompt(target_tokens,tpw,rng): return _to_target(target_tokens,TokenCounter(tpw),rng,"CALL "+uuid.uuid4().hex+" ")
def build_user_message(n_tokens,tpw,rng): return _to_target(n_tokens,TokenCounter(tpw),rng)
