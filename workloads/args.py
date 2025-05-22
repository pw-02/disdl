from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class DisDLConfig:
    name: str
    cache_host:str
    cache_port:int
    grpc_server:str
    ssl_enabled:bool
    use_compression:bool
    num_workers:int

@dataclass
class CoorDLConfig:
    name: str
    cache_host:str
    cache_port:int
    syncronized_mode:bool
    ssl_enabled:bool
    shuffle:bool
    drop_last:bool
    use_compression:bool
    num_workers:int

@dataclass
class JobConfig:
    job_id: int
    gpu_id: int
    model_name:str
    dataset_name:str
    dataset_dir:str
    num_classes:int
    learning_rate: float
    batch_size: int
    seed: int = None
    weight_decay: Optional[float] = None
    sim_gpu_time: Optional[float] = None
    max_training_time_sec: Optional[float] = None
    max_training_steps: Optional[float] = None
    max_epochs: Optional[int] = None
    precision: Optional[str] = None

@dataclass
class FullConfig:
    log_dir: str
    log_interval: int
    dataloader_name: str
    dataset_name: str
    accelerator: str
    simulation_mode: bool
    job: JobConfig
    devices: int
    dataloader: Optional[Union[DisDLConfig, CoorDLConfig]]
    num_jobs: int = 1
    checkpoint_frequency: Optional[int] = None
    
    # add other fields as needed
