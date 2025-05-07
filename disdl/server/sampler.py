from typing import  Dict, Sized
from itertools import cycle
import random
from collections import OrderedDict


class PartitionedBatchSampler():
    def __init__(self, num_files:Sized, batch_size, num_partitions = 10,  drop_last=False, shuffle=True):
        self.num_files = num_files
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        partitions = self._partition_indices(num_partitions) # List of partitions (each a list of indices)
        self.partitions_cycle = cycle(enumerate(partitions))  # Track partition index
        self.num_partitions = len(partitions)
        self.batches_per_epoch = self.calc_num_batchs_per_epoch()

        # Initialize epoch tracking
        self.current_epoch = 1
        self.current_idx = 0
        self.processed_partitions = 0  # Track processed partitions in the current epoch
        # Start with the first partition
        self.active_partition_idx, self.active_partition = next(self.partitions_cycle)
        self.sampler = self._create_sampler(self.active_partition)

    def _create_sampler(self, partition):
        """Create a new sampler based on the shuffle setting."""
        if self.shuffle:
            return iter(random.sample(partition, len(partition)))  # Random order
        else:
            return iter(partition)  # Sequential order
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Generate a mini-batch from the current partition, switching partitions when needed."""
        sampled_indices = []

        while len(sampled_indices) < self.batch_size:
            try:
                sampled_indices.append(next(self.sampler))  # Get an index from the partition
            except StopIteration:
                if not self.drop_last and sampled_indices:
                    # return self.generate_batch(sampled_indices)  # Return smaller batch if drop_last=False
                    self.current_idx += 1
                    return sampled_indices, self.current_epoch, self.active_partition_idx+1, self.current_idx
                
                # Move to the next partition
                self.processed_partitions += 1
                if self.processed_partitions == self.num_partitions:
                    self.current_epoch += 1  # Full epoch completed
                    self.processed_partitions = 0  # Reset for the next epoch
                    self.current_idx = 0  # Reset for the next epoch
                    # print(f"Epoch {self.current_epoch} completed!")  # Notify when an epoch ends

                self.active_partition_idx, self.active_partition = next(self.partitions_cycle)
                self.sampler = self._create_sampler(self.active_partition)
                continue  # Restart batch sampling from new partition

        # return self.generate_batch(sampled_indices)
        self.current_idx += 1
        return sampled_indices, self.current_epoch, self.active_partition_idx+1, self.current_idx
    
    # def generate_batch(self, batch_indices):
    #     next_batch = Batch(batch_indices, self.current_epoch, self.active_partition_idx+1, self.current_idx)
    #     self.current_idx += 1  # Increment the batch index for the next batch
    #     return next_batch

    def _partition_indices(self, num_partitions):
    # Initialize a list to hold the partitions
        indices = list(range(self.num_files))  # Create a list of indices [0, 1, ..., num_files - 1]
        if self.shuffle:
            random.shuffle(indices)  # Shuffle the indices once

        # Split into roughly equal partitions
        partition_size = self.num_files // num_partitions
        partitions = [indices[i * partition_size : (i + 1) * partition_size] for i in range(num_partitions)]

        # Add remaining indices to the last partition (if num_files is not evenly divisible)
        remainder = self.num_files % num_partitions
        for i in range(remainder):
            partitions[i].append(indices[num_partitions * partition_size + i])

        total_files = sum(len(samples) for samples in partitions)
        # #print number files in each partition
        # for i, partition in enumerate(partitions):
        #     print(f"Partition {i}: {len(partition)} files")
        assert total_files == self.num_files

        return partitions
    
    def calc_num_batchs_per_partition(self):
        # Calculate the number of batches
        if self.drop_last:
            return len(self.active_partition) // self.batch_size
        else:
            return (len(self.active_partition) + self.batch_size - 1) // self.batch_size
    
    def calc_num_batchs_per_epoch(self):
        # Calculate the number of batches
        if self.drop_last:
            return self.num_files // self.batch_size
        else:
            return (self.num_files + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    from batch import Batch, BatchSet
    sampler = PartitionedBatchSampler(100, 
                                      batch_size=2, 
                                      num_partitions=5, 
                                      drop_last=False, 
                                      shuffle=True)
    
    epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()
    
    for _ in range(50):
        next_batch:Batch = next(sampler)
        print(f"Batch {next_batch.batch_id} with {len(next_batch.indices)} samples")
        active_epoch_idx = next_batch.epoch_idx
        active_partition_idx = next_batch.partition_idx
        active_batch_set_id = next_batch.batch_set_id

        # Get or create partition dictionary for the current epoch
        partition_batches = epoch_partition_batches.setdefault(
            active_epoch_idx, OrderedDict()
        )

        # Get or create BatchSet for the current partition
        batch_set = partition_batches.setdefault(
            active_partition_idx, BatchSet(active_batch_set_id)
        )
         # Insert the batch into the batch set
        batch_set.batches[next_batch.batch_id] = next_batch

    pass