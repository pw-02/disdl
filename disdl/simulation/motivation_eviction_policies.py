import heapq

class Cache:
    def __init__(self, total_batches, num_jobs):
        self.cache = {}  # Stores {batch_id: count of jobs that processed it}
        self.total_batches = total_batches
        self.num_jobs = num_jobs  # Number of jobs in the system
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_log = []  # Store cache size over time

    def calculate_hit_ratio(self):
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / total_accesses if total_accesses > 0 else 0


    def check_cache(self, batch):
        if batch in self.cache:
            self.cache_hits += 1
            return True
        else:
            self.cache_misses += 1
            return False

    def add_batch(self, batch):
        if batch not in self.cache:
            self.cache[batch] = 0
        else:
            self.cache[batch] += 1

    def mark_processed(self, batch, current_time):
        if batch in self.cache:
            self.cache[batch] += 1
            if self.cache[batch] == self.num_jobs:
                del self.cache[batch]  # Evict the batch
                print(f"Batch {batch} evicted from cache.")
        self.cache_size_log.append((current_time, len(self.cache)))  # Track cache size over time


class Job:
    def __init__(self, id, speed, cache):
        self.id = id
        self.speed = speed  # Time taken per batch (in seconds)
        self.cache:Cache = cache
        self.current_batch = 1  # Start processing from batch 1

    def process_batch(self, current_time):
        if self.current_batch > self.cache.total_batches:
            return  # Stop processing if all batches are done

        if self.cache.check_cache(self.current_batch):
            print(f"Time {current_time}s: Job {self.id} processed batch {self.current_batch} from cache.")
        else:
            print(f"Time {current_time}s: Job {self.id} added batch {self.current_batch} to cache.")
            self.cache.add_batch(self.current_batch)

        self.cache.mark_processed(self.current_batch, current_time)

        # Schedule the next batch event
        next_event_time = current_time + self.speed
        heapq.heappush(event_queue, (next_event_time, self.id))
        self.current_batch += 1  # Move to the next batch

# Initialize Simulation
num_batches = 100
shared_cache = Cache(num_batches, num_jobs=2)
job_speeds = [1, 10]  # Speeds of the two jobs
jobs = []
event_queue = []

for i, speed in enumerate(job_speeds):
    new_job = Job(id=i+1, speed=speed, cache=shared_cache)
    jobs.append(new_job)
    heapq.heappush(event_queue, (new_job.speed, new_job.id))

# Run the simulation
while event_queue:
    current_time, job_id = heapq.heappop(event_queue)
    job:Job = next(j for j in jobs if j.id == job_id)
    job.process_batch(current_time)

#toal duration of simulation
# total_duration = max(shared_cache.cache_size_log, key=lambda x: x[0])[0]
print(f"Total duration of simulation: {current_time}s")
print(f"Cache hit ratio: {shared_cache.calculate_hit_ratio() * 100:.2f}%")
max_cache_size = max(shared_cache.cache_size_log, key=lambda x: x[1])
print(f"Max cache size reached: {max_cache_size[1]} at time {max_cache_size[0]}s")
