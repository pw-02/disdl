import heapq
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Cache:
    def __init__(self):
        self.data = {}  # Stores cached batches
        self.batch_ref_count = {}  # Tracks when a batch can be removed
        self.cache_hits = 0
        self.cache_misses = 0

    def access(self, batch_key, job_id):
        if batch_key in self.data:
            self.cache_hits += 1
            logger.debug(f"Cache hit: Job {job_id} accessed {batch_key}")
        else:
            self.cache_misses += 1
            self.data[batch_key] = set()
            logger.debug(f"Cache miss: Job {job_id} added {batch_key} to cache")
        
        self.data[batch_key].add(job_id)
        self.batch_ref_count[batch_key] = self.batch_ref_count.get(batch_key, 0) + 1
    
    def remove_batch(self, batch_key):
        if self.batch_ref_count.get(batch_key, 0) > 0:
            self.batch_ref_count[batch_key] -= 1
        
        if self.batch_ref_count.get(batch_key, 0) == 0:
            self.data.pop(batch_key, None)
            self.batch_ref_count.pop(batch_key, None)
            logger.debug(f"Removed batch {batch_key} from cache")

class Job:
    def __init__(self, job_id, speed, num_epochs, batches_per_epoch):
        self.job_id = job_id
        self.speed = speed
        self.num_epochs_completed = 0
        self.total_epochs_required = num_epochs
        self.batches_per_epoch = batches_per_epoch
        self.current_epoch = 0
        self.current_batch = 0

    def next_event(self, active_epoch):
        """Determines the next event (epoch, batch) for the job."""
        next_epoch, next_batch = self.current_epoch, self.current_batch + 1
        if next_batch >= self.batches_per_epoch:
            next_epoch = active_epoch
            next_batch = 0
            self.num_epochs_completed += 1
        return next_epoch, next_batch

class Simulation:
    def __init__(self, num_jobs, batches_per_epoch, num_epochs, job_speeds):
        self.event_queue = []
        self.cache = Cache()
        self.jobs = [Job(i, job_speeds[i], num_epochs, batches_per_epoch) for i in range(num_jobs)]
        self.active_epochs = {0}  # Start with epoch 0
        
        # Initialize event queue
        for job in self.jobs:
            heapq.heappush(self.event_queue, (0, job.job_id, 0, 0))

    def run(self):
        while self.event_queue:
            time, job_id, epoch, batch = heapq.heappop(self.event_queue)
            job = self.jobs[job_id]
            batch_key = (epoch, batch)
            
            # Cache access
            self.cache.access(batch_key, job_id)
            
            # Process batch
            job.current_epoch, job.current_batch = epoch, batch
            self.cache.remove_batch(batch_key)
            logger.debug(f"Time {time:.2f}: Job {job_id} processed batch {batch_key}")
            
            # Determine next epoch and batch
            if job.num_epochs_completed < job.total_epochs_required:
                active_epoch = max(j.current_epoch for j in self.jobs)  # Follow fastest job
                next_epoch, next_batch = job.next_event(active_epoch)
                
                # Track active epochs
                self.active_epochs.add(next_epoch)
                
                # Schedule next event
                next_time = time + random.uniform(1, job.speed)
                heapq.heappush(self.event_queue, (next_time, job_id, next_epoch, next_batch))
        
        logger.info("Simulation complete.")

# Example usage
dataset_size = 1281167
batch_size = 128
batches_per_epoch = dataset_size // batch_size
num_epochs = 90
job_speeds = [2, 3]  # Different speeds for each job

sim = Simulation(len(job_speeds), batches_per_epoch, num_epochs, job_speeds)
sim.run()
