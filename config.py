"""Configuration for the GPU-agnostic voice inference load simulator."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import List

@dataclass
class Config:
    base_url: str = "http://localhost:30000/v1"
    model: str = "google/gemma-4-12B-it"
    api_key: str = "EMPTY"
    label: str = "benchmark"
    prompt_sharing: str = "unique"  # shared, unique, multi-template, partial-miss
    system_prompt_tokens: int = 18_000  # legacy alias
    shared_prefix_tokens: int = 18_000
    customer_context_tokens: int = 0
    user_tokens: int = 30
    output_tokens: int = 100
    output_tokens_min: int = 100
    output_tokens_max: int = 100
    prompt_groups: int = 1
    prompt_group_proportions: List[float] = field(default_factory=list)
    cold_miss_percent: float = 0.0
    token_tolerance: int = 64
    tokenizer_local_only: bool = False
    turns: int = 4
    three_turn_probability: float = 0.0
    first_turn_mode: str = "static_first_turn"
    static_response_tokens: int = 40
    warm_turn1_max_tokens: int = 1
    customer_context_each_turn: bool = False
    ignore_eos: bool = True
    temperature: float = 0.0
    timing_mode: str = "fixed-gap"
    inter_turn_gap_s: tuple = (5.0, 7.0)
    turn_schedule_3: List[float] = field(default_factory=lambda: [5.,55.,105.])
    turn_schedule_4: List[float] = field(default_factory=lambda: [5.,35.,70.,105.])
    schedule_jitter_s: float = 0.0
    trace_path: str | None = None
    mean_call_duration_s: float = 120.0
    arrival_rates: List[float] = field(default_factory=lambda: [.1,.2,.5,1.])
    step_duration_s: float = 90.0
    warmup_s: float = 20.0
    drain_s: float = 180.0
    cooldown_s: float = 8.0
    max_concurrent_calls: int = 2000
    max_recorded_requests: int = 200_000
    ttft_slo_ms: float = 300.0
    itl_slo_ms: float = 80.0
    min_knee_samples: int = 20
    max_error_rate: float = .02
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    max_connections: int = 4000
    out_dir: str = "results"
    seed: int = 1234
    def __post_init__(self):
        if self.prompt_sharing == "unique" and self.shared_prefix_tokens == 18_000:
            self.shared_prefix_tokens = self.system_prompt_tokens
    def to_dict(self) -> dict:
        d=asdict(self); d["inter_turn_gap_s"]=list(self.inter_turn_gap_s); d.pop("api_key",None); return d

PRESETS = {
 "voice-production": dict(label="voice-production",prompt_sharing="shared",shared_prefix_tokens=16_320,system_prompt_tokens=16_320,customer_context_tokens=680,user_tokens=100,output_tokens_min=50,output_tokens_max=100,output_tokens=75,turns=4,three_turn_probability=.5,first_turn_mode="llm_first_turn",timing_mode="scheduled",turn_schedule_3=[5.,55.,105.],turn_schedule_4=[5.,35.,70.,105.],mean_call_duration_s=120.,warmup_s=180.,step_duration_s=900.,drain_s=180.,arrival_rates=[.87,1.74,2.48]),
 "h100": dict(label="H100",prompt_sharing="shared",shared_prefix_tokens=16_320,system_prompt_tokens=16_320,customer_context_tokens=680,user_tokens=100,output_tokens_min=50,output_tokens_max=100,output_tokens=75,three_turn_probability=.5,first_turn_mode="llm_first_turn",timing_mode="scheduled",arrival_rates=[.87,1.74,2.48],warmup_s=180.,step_duration_s=900.,drain_s=180.),
 "h200": dict(label="H200",prompt_sharing="shared",shared_prefix_tokens=16_320,system_prompt_tokens=16_320,customer_context_tokens=680,user_tokens=100,output_tokens_min=50,output_tokens_max=100,output_tokens=75,three_turn_probability=.5,first_turn_mode="llm_first_turn",timing_mode="scheduled",arrival_rates=[1.74,3.47,4.96],warmup_s=180.,step_duration_s=900.,drain_s=180.),
 "l40s":dict(label="L40S"), "smoke":dict(arrival_rates=[.2,.5,1.],step_duration_s=15.,warmup_s=3.,drain_s=2.,shared_prefix_tokens=2000,system_prompt_tokens=2000), }
