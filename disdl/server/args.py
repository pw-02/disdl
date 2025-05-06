from dataclasses import dataclass

@dataclass
class DisDLArgs:
    cache_address:str
    use_prefetching:bool
    prefetch_lambda_name:str
    prefetch_cost_cap_per_hour:float
    prefetch_simulation_time:float
    workload:str