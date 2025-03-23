
from batch_manager import CentralBatchManager
from args import DisDLArgs
from dataset import ImageNetDataset
from batch import Batch
import time
import heapq
import numpy as np
import time
import logging


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Event:
    def __init__(self, timestamp, event_type, job_id):
        self.timestamp = timestamp
        self.event_type = event_type  # 'batch_request', 'eviction', etc.
        self.job_id = job_id
   
    def __lt__(self, other):
        return self.timestamp < other.timestamp  # Min-heap sorting by time

class NextEventSimulator:
    def __init__(self, 
                 batch_manager, 
                 job_speeds, 
                 batch_size,
                 batches_per_epoch,
                 epochs_per_job):
        
        self.batch_manager:CentralBatchManager = batch_manager
        self.job_speed = job_speeds  # List of speeds for each job
        self.event_queue = []
        self.current_time = 0.0
        self.job_last_batch_time = {i: 0.0 for i in range(len(job_speeds))}
        self.batches_in_system_over_time = {}
        #dict to count number of epochs processed by each job
        self.job_epoch_count = {i: 0 for i in range(len(job_speeds))}
        #dict to store the current epoch of each job
        self.job_current_epoch = {i: 0 for i in range(len(job_speeds))}
        self.epochs_per_job = epochs_per_job
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.job_performance = {i: 0 for i in range(len(job_speeds))}
    
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
        
        event:Event = heapq.heappop(self.event_queue)
        self.current_time = event.timestamp

        if event.event_type == "batch_request":
            self.handle_batch_request(event.job_id)

        return event

    def handle_batch_request(self, job_id):
        """Simulate a job requesting a batch."""
        batch = self.batch_manager.get_next_batch_for_job(job_id)
        if batch is not None:
            self.batches_in_system_over_time[self.current_time] = self.batch_manager.total_active_batches()
            epoch_idx = batch.epoch_idx
            #check if the job started a new epoch
            if epoch_idx > self.job_current_epoch[job_id]:
                self.job_epoch_count[job_id] += 1
                self.job_current_epoch[job_id] = epoch_idx
                
                if self.job_epoch_count[job_id] < self.epochs_per_job:
                    logger.info(f"Time {self.current_time:.2f}: Job {job_id} started epoch {epoch_idx}. Batches in system: {list(self.batches_in_system_over_time.values())[-1]}")
            
            # Schedule the next batch request unless job has processed the specified number of epochs
            if self.job_epoch_count[job_id] < self.epochs_per_job:
                next_request_time = self.current_time + self.job_speed[job_id]
                # logger.debug(f"Time {self.current_time:.2f}: Job {job_id} has processed {batch.batch_id}. Batches in system: {self.num_batches_in_system_over_time[-1]}")
                self.schedule_event(Event(next_request_time, "batch_request", job_id))
            else:
                logger.info(f"Time {self.current_time:.2f}: Job {job_id} has processed {self.epochs_per_job} epochs and will not request more batches.")
                #print job completion te and throughput
                job_completion_time = self.current_time 
                throughput = self.batch_size * self.batches_per_epoch * self.epochs_per_job / job_completion_time
                compute_cost = self.calc_compute_costs(job_completion_time, single_job = True)
                logger.info(f"Job {job_id} completed {self.epochs_per_job} epochs in {job_completion_time/3600:.2f} mins. Throughput: {throughput:.2f} samples/sec")
                self.job_performance[job_id] = {'job_training_time': job_completion_time/3600, 'throughput': throughput, 'compute_cost': compute_cost}
        else:
            print(f"[{self.current_time:.3f}] Job {job_id} failed to fetch a batch (cache miss or empty).")

    def calc_compute_costs(self, training_time_seconds, ec2_hourly_cost=12.24, single_job=False):
        if single_job:
            return ec2_hourly_cost * (training_time_seconds / 3600) / len(self.job_speed) # Cost in USD
        else:
            return ec2_hourly_cost * (training_time_seconds / 3600) # Cost in USD

    def compute_aws_lambda_cache_costs(self, num_warmup_requets):
        #total batches processed by all jobs
        cost_per_prefetch = 0.000110972
        num_prefetch_batches = self.epochs_per_job * self.batches_per_epoch * self.epochs_per_job
        prefetch_cost = num_prefetch_batches * cost_per_prefetch

        avg_lambda_cost_per_request = 0.00000830198
        total_batches_per_job = sum([self.epochs_per_job * self.batches_per_epoch for _ in range(len(self.job_speed))])
        get_put_request_cost = (total_batches_per_job + num_prefetch_batches) * avg_lambda_cost_per_request
        warm_up_cost = num_warmup_requets * avg_lambda_cost_per_request

        total_cost = prefetch_cost + get_put_request_cost + warm_up_cost

        result = {
            'prefetch_cost': prefetch_cost,
            'get_put_request_cost': get_put_request_cost,
            'warm_up_cost': warm_up_cost,
            'total_cost': total_cost}
        return result
    
    def run_simulation(self,):
        self.initialize_jobs()   
        while self.event_queue:
            self.process_next_event()

    def report_results(self):
        print(f"Max number of batches in system: {max(list(self.batches_in_system_over_time.values()))}")
        print(f"Min number of batches in system: {min(list(self.batches_in_system_over_time.values()))}")
        print(f"Mean number of batches in system: {np.mean(list(self.batches_in_system_over_time.values()))}")
        for job_id, performance in self.job_performance.items():
            print(f"Job {job_id} completed {self.epochs_per_job} epochs in {performance['job_training_time']:.2f} seconds. Throughput: {performance['throughput']:.2f} samples/sec. Cost: {performance['compute_cost']:.2f} USD")

        #aggreate throughput
        total_throughput = sum([performance['throughput'] for performance in self.job_performance.values()])
        print(f"Total throughput: {total_throughput:.2f} samples/sec")

        #lamnda cache costs
        lambda_costs = self.compute_aws_lambda_cache_costs(num_warmup_requets=np.mean(list(self.batches_in_system_over_time.values())))
        total_compute_cost = sum([performance['compute_cost'] for performance in self.job_performance.values()])
        print(f"Total compute cost: {total_compute_cost:.2f}")
        print(f"Total lambda cost: {lambda_costs['total_cost']}")
        print(f"   Prefethc cost: {lambda_costs['prefetch_cost']}")
        print(f"   Get/Put request cost: {lambda_costs['get_put_request_cost']}")

        total_cost = total_compute_cost + lambda_costs['total_cost']
        print(f"Total cost: {total_cost:.2f}")

# Example Usage
job_speeds = [0.137794227,0.14272167,0.351509787,0.519805225]

args:DisDLArgs = DisDLArgs(
            batch_size = 128,
            num_dataset_partitions = 2,
            lookahead_steps = 1,
            shuffle = False,
            drop_last = False,
            workload = 'vision',
            serverless_cache_address = None,
            use_prefetching = False,
            use_keep_alive = False,
            prefetch_lambda_name = 'CreateVisionTrainingBatch',
            prefetch_cost_cap_per_hour=None,
            cache_keep_alive_timeout = 60 * 3, # 3 minutes
            prefetch_simulation_time = None,
            evict_from_cache_simulation_time = None)
cifar= 's3://sdl-cifar10/test/'
imagenet = 's3://imagenet1k-sdl/train/'
# dataset = ImageNetDataset(dataset_location='s3://imagenet1k-sdl/train/')
dataset = ImageNetDataset(dataset_location=imagenet)
num_samples  = len(dataset)
num_batches = num_samples // args.batch_size
batch_manager = CentralBatchManager(dataset=dataset, args=args)

simulator = NextEventSimulator(
   batch_manager=batch_manager,
   job_speeds= job_speeds,
   batch_size = args.batch_size,
   batches_per_epoch = num_batches,
   epochs_per_job = 1)

simulator.run_simulation()
simulator.report_results()
#plot the number of batches in the system over time
# import matplotlib.pyplot as plt
# plt.plot(list(simulator.batches_in_system_over_time.keys()), list(simulator.batches_in_system_over_time.values()))
# plt.xlabel('Time')
# plt.ylabel('Number of Batches in System')
# plt.title('Number of Batches in System Over Time')
# plt.show()

#0.00003125