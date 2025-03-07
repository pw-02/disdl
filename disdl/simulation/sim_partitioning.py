
from batch_manager import CentralBatchManager
from args import DisDLArgs
from dataset import Dataset
from batch import Batch
import time
import heapq
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
#set to info mode
logger.setLevel(logging.INFO)

class Event:
    def __init__(self, timestamp, event_type, job_id):
        self.timestamp = timestamp
        self.event_type = event_type  # 'batch_request', 'eviction', etc.
        self.job_id = job_id

    def __lt__(self, other):
        return self.timestamp < other.timestamp  # Min-heap sorting by time
    
class NextEventSimulator:
    def __init__(self, batch_manager, job_speed):
        self.batch_manager:CentralBatchManager = batch_manager
        self.job_speed = job_speed  # List of speeds for each job
        self.event_queue = []
        self.current_time = 0.0
        self.job_last_batch_time = {i: 0.0 for i in range(len(job_speed))}
        self.num_batches_in_system_over_time = []
    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def initialize_jobs(self):
        """Schedules the first batch request for each job."""
        for job_id, speed in enumerate(self.job_speed):
            first_request_time = np.random.exponential(speed)
            self.schedule_event(Event(first_request_time, "batch_request", job_id))

    def process_next_event(self):
        if not self.event_queue:
            return None
        
        event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp

        if event.event_type == "batch_request":
            self.handle_batch_request(event.job_id)

        return event

    def handle_batch_request(self, job_id):
        """Simulate a job requesting a batch."""
        batch = self.batch_manager.get_next_batch_for_job(job_id)
        if batch is not None:
            # print(f"[{self.current_time:.3f}] Job {job_id} fetched a batch.")
            
            # Schedule the next batch request based on the job's speed
            self.num_batches_in_system_over_time.append(self.batch_manager.total_active_batches())
            next_request_time = self.current_time + self.job_speed[job_id]
            self.schedule_event(Event(next_request_time, "batch_request", job_id))
        else:
            print(f"[{self.current_time:.3f}] Job {job_id} failed to fetch a batch (cache miss or empty).")

    def run_simulation(self, max_time=100):
        self.initialize_jobs()
        while self.event_queue and self.current_time < max_time:
            self.process_next_event()

# Example Usage
job_speeds = [0.1, 0.001]

args:DisDLArgs = DisDLArgs(
            batch_size = 100,
            num_dataset_partitions = 10,
            lookahead_steps = 10,
            shuffle = False,
            drop_last = False,
            workload_kind = 'vision',
            serverless_cache_address = None,
            use_prefetching = False,
            use_keep_alive = False,
            prefetch_lambda_name = 'CreateVisionTrainingBatch',
            prefetch_cost_cap_per_hour=None,
            cache_keep_alive_timeout = 60 * 3, # 3 minutes
            prefetch_simulation_time = None,
            evict_from_cache_simulation_time = None)

dataset = Dataset(data_dir='s3://sdl-cifar10/test/')
batch_manager = CentralBatchManager(dataset=dataset, args=args)

simulator = NextEventSimulator(batch_manager, job_speeds)
simulator.run_simulation(max_time=60)
print(f"Max number of partitions in system: {max(simulator.num_batches_in_system_over_time)}")
