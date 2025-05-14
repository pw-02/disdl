from dataclasses import dataclass
from typing import List

@dataclass
class DatasetConfig:
    name: str
    storage_backend: str
    batch_size: int
    shuffle: bool
    drop_last: bool
    num_partitions: int
    prefetch_lambda_name: str

@dataclass
class DisDLArgs:
    available_datasets: List[DatasetConfig]
    cache_address: str
    enable_prefetching: bool
    prefetch_cost_cap_per_hour: float
    # prefetch_simulation_time: float = None
