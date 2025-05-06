from dataclasses import dataclass

@dataclass
class DisDLArgs:
    cache_address:str
    enable_prefetching:bool
    prefetch_cost_cap_per_hour:float
    prefetch_simulation_time:float
    workload:str