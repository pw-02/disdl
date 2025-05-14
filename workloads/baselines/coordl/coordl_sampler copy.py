from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized, Tuple
import hashlib
import torch
import random

class CoorDLBatchSampler(BatchSampler):
    def __init__(self, 
                    data_source: Sized,
                    batch_size: int,
                    job_idx: int,
                    num_jobs: int,
                    drop_last: bool = False,
                    shuffle: bool = True,
                    seed: Optional[int] = None):
        
        self.job_idx = job_idx
        self.num_jobs = num_jobs
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Choose the appropriate sampler based on shuffle argument
        sampler = RandomSampler(data_source) if shuffle else SequentialSampler(data_source)
        super().__init__(sampler, batch_size, drop_last)
        # Initialize the base BatchSampler with the chosen sampler
        #         
    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
        # Step 1: Get all batches from the base class
        batches = list(super().__iter__())
         # Step 2: Split batches into producer groups
        producer_batches = [batches[i::self.num_jobs] for i in range(self.num_jobs)]

        # Step 3: Create coordinated (rotated) batch access order
        ordered_batches = []
        for round_idx in range(len(producer_batches[0])):
            for offset in range(self.num_jobs):
                prod_id = (self.job_idx + offset) % self.num_jobs
                if round_idx < len(producer_batches[prod_id]):
                    ordered_batches.append(producer_batches[prod_id][round_idx])
        
        # Step 4: Yield each batch with a hash-based batch ID
        for batch_indices in ordered_batches:
            batch_id = hashlib.md5(str(batch_indices).encode()).hexdigest()
            yield (batch_id, batch_indices)

    def __len__(self) -> int:
        return super().__len__()

# Example usage
if __name__ == "__main__":
    from torch.utils.data import SequentialSampler

    # Example dataset size
    dataset_size = 10
    num_jobs = 4
    samplers = []
    for job_idx in range(1, num_jobs + 1):
        batch_sampler = CoorDLBatchSampler(
        data_source=range(dataset_size),
        batch_size=1,
        job_idx=job_idx,
        num_jobs=num_jobs,
        shuffle=False,
        drop_last=True,
        seed=123)
        samplers.append(batch_sampler)
        # Create a batch sampler for each job inde
        indicies =[]
        for batch_id, batch_indices in batch_sampler:
            indicies.append(batch_indices)
        print(f"Job: {job_idx}, Batch Indices: {indicies}")


    # # Iterate over batches and print batch IDs and indices
    # for batch_sampler in samplers:
    #     for batch_id, batch_indices in batch_sampler:
    #         print(f"Batch ID: {batch_id}, Batch Indices: {batch_indices}")



    # # Example usage of BatchSamplerWithID with shuffling
    # batch_sampler_with_id = TensorSocketSampler(data_source=range(dataset_size), batch_size=10, drop_last=False, shuffle=True, seed=42)

    # # Iterate over batches and print batch IDs and indices
    # for batch_id, batch_indices in batch_sampler_with_id:
    #     print(f"Batch ID: {batch_id}, Batch Indices: {batch_indices}")
