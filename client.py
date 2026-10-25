"""Streaming OpenAI client. Chunk timing is deliberately not token timing."""
from __future__ import annotations
import json,time
from dataclasses import dataclass,field
from typing import List,Optional
import httpx
@dataclass
class TurnResult:
    ok:bool; assistant_text:str=""; ttft_ms:float=0.; transport_first_ms:float=0.; itls_ms:List[float]=field(default_factory=list) # compatibility: content chunk gaps
    chunk_gaps_ms:List[float]=field(default_factory=list); n_output:int=0; completion_tokens:Optional[int]=None
    prompt_tokens:int=0; cached_tokens:Optional[int]=None; fresh_tokens:Optional[int]=None; total_s:float=0.; error:Optional[str]=None; http_status:Optional[int]=None; finish_reason:Optional[str]=None; content_timestamps:List[float]=field(default_factory=list)
    @property
    def tpot_ms(self):
        # Only accurate if server reports completion count AND chunks map 1:1; do not claim it otherwise.
        return None
async def stream_chat(client:httpx.AsyncClient,model:str,messages:list,max_tokens:int,ignore_eos:bool,temperature:float)->TurnResult:
    payload={"model":model,"messages":messages,"max_tokens":max_tokens,"temperature":temperature,"stream":True,"stream_options":{"include_usage":True},"return_cached_tokens_details":True}
    if ignore_eos: payload["ignore_eos"]=True
    start=time.monotonic(); first_event=None; first_content=None; last_content=None; chunks=[]; gaps=[]; usage={}; finish=None
    try:
      async with client.stream("POST","/chat/completions",json=payload) as resp:
        if resp.status_code!=200:
          body=(await resp.aread()).decode("utf-8","ignore")[:200]; return TurnResult(False,total_s=time.monotonic()-start,error=f"HTTP {resp.status_code}: {body}",http_status=resp.status_code)
        async for line in resp.aiter_lines():
          if not line or not line.startswith("data:"): continue
          now=time.monotonic(); first_event=first_event or now; data=line[5:].strip()
          if data=="[DONE]": break
          try: chunk=json.loads(data)
          except json.JSONDecodeError: continue
          if chunk.get("usage"): usage=chunk["usage"]
          choices=chunk.get("choices") or []
          if not choices: continue
          choice=choices[0]; finish=choice.get("finish_reason") or finish; delta=choice.get("delta") or {}
          # Content only: role/reasoning/tool metadata must not establish voice TTFT.
          content=delta.get("content")
          if isinstance(content,str) and content:
            if first_content is None: first_content=now
            elif last_content is not None: gaps.append((now-last_content)*1000.)
            last_content=now; chunks.append(content)
    except (httpx.HTTPError,httpx.StreamError,TimeoutError) as e:
      return TurnResult(False,total_s=time.monotonic()-start,error=str(e))
    total=time.monotonic()-start
    if first_content is None: return TurnResult(False,total_s=total,error="stream ended without assistant content",finish_reason=finish)
    from prompt import measured_prompt_tokens,cached_prompt_tokens
    pt=measured_prompt_tokens(usage); cached=cached_prompt_tokens(usage)
    completion=(usage or {}).get("completion_tokens")
    return TurnResult(True,assistant_text="".join(chunks),ttft_ms=(first_content-start)*1000.,transport_first_ms=((first_event-start)*1000. if first_event else 0.),itls_ms=gaps,chunk_gaps_ms=gaps,n_output=(int(completion) if completion is not None else 0),completion_tokens=(int(completion) if completion is not None else None),prompt_tokens=pt,cached_tokens=cached,fresh_tokens=(max(0,pt-cached) if cached is not None else None),total_s=total,http_status=200,finish_reason=finish,content_timestamps=[])
