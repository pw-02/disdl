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
        self.owned_batch_ids = set()

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
        all_batches = list(super().__iter__())

        # Step 1: Split into contiguous producer groups
        total_batches = len(all_batches)
        batches_per_job = total_batches // self.num_jobs
        remainder = total_batches % self.num_jobs

        job_to_batches = []
        cursor = 0
        for i in range(self.num_jobs):
            num = batches_per_job + (1 if i < remainder else 0)
            job_to_batches.append(all_batches[cursor:cursor + num])
            cursor += num

        # Step 2: Construct this job's access order
        ordered_batches = []
        max_len = max(len(group) for group in job_to_batches)
        for r in range(max_len):
            for offset in range(self.num_jobs):
                prod_id = (self.job_idx + offset) % self.num_jobs
                if r < len(job_to_batches[prod_id]):
                    ordered_batches.append(job_to_batches[prod_id][r])

        # Step 3: Yield batch_id and batch_indices
        for batch_indices in ordered_batches:
            batch_id = hashlib.md5(str(batch_indices).encode()).hexdigest()
            if batch_indices in job_to_batches[self.job_idx]:
                yield batch_id, batch_indices, True
            else:
                yield batch_id, batch_indices, False

    def __len__(self) -> int:
        return super().__len__()

# Example usage
if __name__ == "__main__":
    from torch.utils.data import SequentialSampler

    # Example dataset size
    dataset_size = 10
    num_jobs = 4
    samplers = []
    for job_idx in range(num_jobs):
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
        for batch_id, batch_indices, owner in batch_sampler:
            indicies.append((batch_indices, owner))
        print(f"Job: {job_idx}, Batch Indices: {indicies}")
