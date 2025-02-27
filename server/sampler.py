from typing import  Dict, Sized
from itertools import cycle
import random
from batch import Batch, BatchSet
from collections import OrderedDict
import hashlib

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
        self.current_epoch = 0
        self.processed_partitions = 0  # Counts how many partitions we've used
        self.current_idx = 0  # Counts how many batches we've generated
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
                    
                    return self.generate_batch(sampled_indices)  # Return smaller batch if drop_last=False
                
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

        return self.generate_batch(sampled_indices)
    
    def generate_batch(self, batch_indices):
        self.current_idx += 1
        batch_id = f"{self.current_epoch}_{self.active_partition_idx}_{self.current_idx}_{self.create_unique_id(batch_indices, 16)}"
        return Batch(batch_indices, batch_id, self.current_epoch, self.active_partition_idx)

    def create_unique_id(self,int_list, length = 32):
        # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in int_list)
        
        # Hash the concatenated string to generate a unique ID
        unique_id = hashlib.md5(id_string.encode()).hexdigest()
        #Truncate the hash to the desired length
        return unique_id[:length]

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
    
    def calc_num_batchs_per_epoch(self):
        # Calculate the number of batches
        if self.drop_last:
            return self.num_files // self.batch_size
        else:
            return (self.num_files + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    
    from dataset import Dataset
    dataset = Dataset(
        data_dir="s3://sdl-cifar10/test/",
        transforms=None,
        max_dataset_size=None,
        use_local_folder=False)
    print(f"Total samples: {len(dataset)}")

    sampler = PartitionedBatchSampler(len(dataset), 
                                      batch_size=100, 
                                      num_partitions=10, 
                                      drop_last=False, 
                                      shuffle=True)
    
    epoch_partition_batches: Dict[int, Dict[int, BatchSet]] = OrderedDict()  #first key is epoch id, second key is partition id, value is the batches

    # Generate initial batches
    for _ in range(10000):
        next_batch:Batch = next(sampler)
        # Ensure epoch exists, initializing with an OrderedDict for partitions
        partition_batches = epoch_partition_batches.setdefault(next_batch.epoch_idx, OrderedDict())
        
         # Ensure partition exists, initializing with a new BatchSet if needed
        partition_batch_set = partition_batches.setdefault(
            next_batch.partition_id, BatchSet(f'{next_batch.epoch_idx}_{next_batch.partition_id}')
        )
        # Store the batch in the BatchSet
        partition_batch_set.batches[next_batch.batch_id] = next_batch
    
    print(epoch_partition_batches)

    # # print(f"Total partitions: {len(sampler.partitions_cycle)}")
    # paths = set()  # Use a set for quick duplicate detection
    # path_list = []  # Maintain the ordered list for debugging

    # for i, batch in enumerate(sampler):
        
    #     for idx in batch.indicies:
    #         path, label = dataset._classed_items[idx]
            
    #         if path in paths:  # Check if path already exists
    #             print(f"Duplicate detected! Total unique paths before duplicate: {len(paths)}")
    #             exit()  # Stop execution immediately
            
    #         paths.add(path)
    #         path_list.append(path)  # Keep track of order (for debugging)
    #     print(f"Batch {i}: {batch.batch_id} with {len(batch.indicies)} samples")
