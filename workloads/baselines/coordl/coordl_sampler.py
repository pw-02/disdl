
import random
import hashlib
from typing import Sized, List
import redis


class CoorDLBatchSampler():
    def __init__(self, num_files:Sized, batch_size, jobid, drop_last=False, shuffle=True,
                cache_address= None,
                 ssl = True,):
        
        self.job_id = jobid
        self.num_files = num_files
        self.file_indices = list(range(num_files))          
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batches_per_epoch = self.calc_num_batchs_per_epoch()
        random.seed(42)
        # Initialize epoch tracking
        self.current_epoch = 0
        self.processed_partitions = 0  # Counts how many partitions we've used
        self.current_idx = 0  # Counts how many batches we've generated
        # Start with the first partition
        if cache_address is not None:
            self.cache_host, self.cache_port = cache_address.split(":")
            self.cache_port = int(self.cache_port)
            self.use_cache = True
        else:
            self.use_cache = False

        self.ssl = ssl
        self.sampler = self._create_sampler(num_files)
        self.batches:List = []
        self.get_batches()
        self.cache_client = None
        pass
    

    def get_batches(self):
        batches_per_epoch = self.batches_per_epoch
        for idx in range(batches_per_epoch):
            batch_indices, batch_id = next(self)
            this_job_fetch = False
            if (idx-self.job_id) % 4 == 0:
                this_job_fetch = True
                # batch_samples = [self._classed_items[i] for i in batch_indices]
            self.batches.append((batch_indices, batch_id, this_job_fetch))

       
    def _create_sampler(self, partition):
        """Create a new sampler based on the shuffle setting."""
        if self.shuffle:
            #seed the random number generator for reproducibility
            random.shuffle(self.file_indices)
            return iter(self.file_indices)  # Random order
        else:
            return iter(self.file_indices)  # Sequential order
    
    def _initialize_cache_client(self):
        """Initialize the cache client."""
        # Placeholder for actual cache client initialization
        # self.cache_client = CacheClient()
        """Initialize Redis cache client if not already connected."""
        if self.cache_client is None:
            if self.ssl:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port, ssl=True)
            else:
                self.cache_client = redis.StrictRedis(host=self.cache_host, port=self.cache_port)


    def __iter__(self):
        while len(self.batches) > 0:
            next_batch = None
            cache_hit = False
            for batch in self.batches:
                batch_indices, batch_id, this_job_fetch = batch
                self._initialize_cache_client()
                if self.cache_client.exists(batch_id): #cached by another job use it
                    next_batch = batch
                    cache_hit = True
                    break
                
                if this_job_fetch:
                    next_batch = batch
                    break
            
            if next_batch is None:
                next_batch = self.batches[0]

            self.batches.remove(next_batch)
            yield  next_batch
    
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
                self.current_epoch += 1  # Full epoch completed
                self.processed_partitions = 0  # Reset for the next epoch
                self.current_idx = 0  # Reset for the next epoch
                self.sampler = self._create_sampler(self.num_files)
                continue  # Restart batch sampling from new partition

        return self.generate_batch(sampled_indices)
    
    def generate_batch(self, batch_indices):
        batch_id = f"{self.current_epoch}_{self.current_idx}_{self.create_unique_id(batch_indices, 16)}"
        self.current_idx += 1
        return batch_indices, batch_id

    def create_unique_id(self,int_list, length = 32):
        # Convert integers to strings and concatenate them
        id_string = ''.join(str(x) for x in int_list)
        
        # Hash the concatenated string to generate a unique ID
        unique_id = hashlib.md5(id_string.encode()).hexdigest()
        #Truncate the hash to the desired length
        return unique_id[:length]

    def calc_num_batchs_per_epoch(self):
        # Calculate the number of batches
        if self.drop_last:
            return self.num_files // self.batch_size
        else:
            return (self.num_files + self.batch_size - 1) // self.batch_size
