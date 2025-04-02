
from batch_manager import CentralBatchManager
from args import DisDLArgs
from dataset import ImageNetDataset
from batch import Batch
import time
import heapq
import numpy as np
import time
import logging
import os
import csv

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)
logger = logging.getLogger("SIMULATOR")
logger.setLevel(logging.INFO)
# logger.handlers= [logging.FileHandler("logfile.log"), logging.StreamHandler()]
logger.__format__ = "%(asctime)s - %(levelname)s - %(message)s"
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
        self.job_progress = {i: 0 for i in range(len(job_speeds))}
        self.unique_bacthes = set()
    
    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def initialize_jobs(self):
        """Schedules the first batch request for each job."""
        for job_id, speed in enumerate(self.job_speed):
            first_request_time = np.random.exponential(speed)
            self.schedule_event(Event(0, "batch_request", job_id))

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
            if batch.batch_id not in self.unique_bacthes:
                    self.unique_bacthes.add(batch.batch_id)
            # Update the number of batches in the system
            self.job_progress[job_id] += 1
            if self.job_progress[job_id] >= self.epochs_per_job * self.batches_per_epoch:
                logger.info(f"Time {self.current_time:.2f}: Job {job_id} has processed {self.epochs_per_job} epochs and will not request more batches.")
                job_completion_time = self.current_time 
                throughput = self.batch_size * self.batches_per_epoch * self.epochs_per_job / job_completion_time
                compute_cost = self.calc_compute_costs(job_completion_time, single_job = True)
                logger.info(f"Job {job_id} completed {self.epochs_per_job} epochs in {job_completion_time/3600:.2f} mins. Throughput: {throughput:.2f} samples/sec")
                self.job_performance[job_id] = {'job_training_time': job_completion_time, 'throughput': throughput, 'compute_cost': compute_cost, 'total_batches': self.job_progress[job_id]}
            else:
                if self.job_progress[job_id] % 50 == 0:
                    self.batches_in_system_over_time[self.current_time] = self.batch_manager.total_active_batches()
                logger.debug(f"Time {self.current_time:.2f}: Job {job_id} has processed {batch.batch_id}.")
                next_request_time = self.current_time + self.job_speed[job_id]
                self.schedule_event(Event(next_request_time, "batch_request", job_id))
        else:
            print(f"[{self.current_time:.3f}] Job {job_id} failed to fetch a batch (cache miss or empty).")

    def calc_compute_costs(self, training_time_seconds, ec2_hourly_cost=12.24, single_job=False):
        if single_job:
            return ec2_hourly_cost * (training_time_seconds / 3600) / len(self.job_speed) # Cost in USD
        else:
            return ec2_hourly_cost * (training_time_seconds / 3600) # Cost in USD
    
    def compute_aws_lambda_cache_costs(self, num_requets, is_prefetch=False):
        #total batches processed by all jobs
        cost_per_prefetch = 0.000110972
        avg_lambda_cost_per_request = 0.00000830198
        if is_prefetch:
            prefetch_cost = num_requets * cost_per_prefetch
            return prefetch_cost
        else:
            total_cost = num_requets * avg_lambda_cost_per_request
            return total_cost
    
    def run_simulation(self,):
        self.initialize_jobs()   
        while self.event_queue:
            self.process_next_event()

    def report_results(self, job_speeds, epochs_per_job,batch_size, partion_size):
        
        jobs_times = [performance['job_training_time'] for performance in self.job_performance.values()]
        mean_batches_in_system = np.mean(list(self.batches_in_system_over_time.values()))
        print(f"Max number of batches in system: {mean_batches_in_system}")
        print(f"Min number of batches in system: {min(list(self.batches_in_system_over_time.values()))}")
        print(f"Mean number of batches in system: {np.mean(list(self.batches_in_system_over_time.values()))}")
        for job_id, performance in self.job_performance.items():
            print(f"Job {job_id} completed {self.epochs_per_job} epochs in {performance['job_training_time']:.2f} seconds. Throughput: {performance['throughput']:.2f} samples/sec. Cost: {performance['compute_cost']:.2f} USD")

        #aggreate throughput
        batches_per_epoch = self.batches_per_epoch
        print(f"Batches per epoch: {batches_per_epoch}")
        
        total_throughput = sum([performance['throughput'] for performance in self.job_performance.values()])
        print(f"Total throughput: {total_throughput:.2f} samples/sec")

        #lamnda cache costs
        # lambda_requests = self.batch_manager.cacl_lamda_invocation_counts()

        # num_prefetches = lambda_requests['num_prefetches']
        # num_warmpup_requests = lambda_requests['num_warmup_requests']
        # num_getset_requests = lambda_requests['num_getset_requests']
        num_prefetches = len(self.unique_bacthes)
        #sum totla batches processed by all jobs
        num_getset_requests = sum([self.epochs_per_job * self.batches_per_epoch for _ in range(len(self.job_speed))])
        num_getset_requests += num_prefetches

        #time in minutes is the max training of the co
        time_in_minutes = max(jobs_times) / 60
        num_warmpup_requests = mean_batches_in_system / time_in_minutes

        prefetch_cost = self.compute_aws_lambda_cache_costs(num_prefetches, is_prefetch=True)
        get_put_request_cost = self.compute_aws_lambda_cache_costs(num_getset_requests)
        warm_up_cost = self.compute_aws_lambda_cache_costs(num_warmpup_requests)
        total_compute_cost = sum([performance['compute_cost'] for performance in self.job_performance.values()])
        total_lambda_cost = prefetch_cost + get_put_request_cost + warm_up_cost
        print(f"Total compute cost: {total_compute_cost:.2f}")
        print(f"Total lambda cost: {total_lambda_cost:.2f}")
        print(f"   Prefecth cost: {prefetch_cost}")
        print(f"   Get/Put request cost: {get_put_request_cost}")
        print(f"   Warm up cost: {warm_up_cost}")
        total_cost = total_compute_cost + total_lambda_cost
        print(f"Total cost: {total_cost:.2f}")

        single_epoch_cost = total_cost / self.epochs_per_job
        single_epoch_throughput = total_throughput / self.epochs_per_job

        #total time
        aggre_time = sum([performance['job_training_time'] for performance in self.job_performance.values()])
        jobs_times = [performance['job_training_time'] for performance in self.job_performance.values()]
        line = {
            'job_speeds': str(job_speeds),
            'epochs_per_job': epochs_per_job,
            'batches_per_epoch': self.batches_per_epoch,
            'batch_size': batch_size,
            'num_partitions': partion_size,
            'epochs_per_job': self.epochs_per_job,
            'max_batches_in_system': max(list(self.batches_in_system_over_time.values())),
            'min_batches_in_system': min(list(self.batches_in_system_over_time.values())),
            'mean_batches_in_system': np.mean(list(self.batches_in_system_over_time.values())),
            'agrregate_time': aggre_time,
            'job_times': str(jobs_times),
            'job_throughputs': str([performance['throughput'] for performance in self.job_performance.values()]),
            'job_num_batches': str([performance['total_batches'] for performance in self.job_performance.values()]),
            'total_throughput': total_throughput,
            'num_prefetches': num_prefetches,
            'prefetch_cost': prefetch_cost,
            'num_getset_requests': num_getset_requests,
            'get_put_request_cost': get_put_request_cost,
            'num_warmup_requests': num_warmpup_requests,
            'warm_up_cost': warm_up_cost,
            'total_lambda_cost': total_lambda_cost,
            'total_compute_cost': total_compute_cost,
            'total_cost': total_cost,
            'single_epoch_cost': single_epoch_cost,
            'single_epoch_throughput': single_epoch_throughput
        }

        #save to csv
        filename = 'simulation_results.csv'
        file_exists = os.path.isfile(filename)

        # #delete file if it exists
        # if file_exists:
        #     os.remove(filename)

        with open(filename, 'a') as f:
            headers = list(line.keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(line)

# Example Usage
# job_speeds = [0.351509787,0.519805225]
job_speeds = [0.137794227,0.14272167,0.351509787,0.519805225]

batches_sizes = [128,64,32,16,8]
num_partitions = [1]

for batch_size in batches_sizes:
    for partion_size in num_partitions:
        args:DisDLArgs = DisDLArgs(
                    batch_size = batch_size,
                    num_dataset_partitions = partion_size,
                    lookahead_steps = 1,
                    shuffle = True,
                    drop_last = True,
                    workload = 'vision',
                    serverless_cache_address = None,
                    use_prefetching = False,
                    use_keep_alive = False,
                    prefetch_lambda_name = 'CreateVisionTrainingBatch',
                    prefetch_cost_cap_per_hour=None,
                    cache_keep_alive_timeout = 60 * 1, # 3 minutes
                    prefetch_simulation_time = 0,    
                    evict_from_cache_simulation_time = None)
        cifar= 's3://sdl-cifar10/test/'
        imagenet = 's3://imagenet1k-sdl/train/'
        epochs_per_job = 10
        # dataset = ImageNetDataset(dataset_location='s3://imagenet1k-sdl/train/')
        dataset = ImageNetDataset(dataset_location=imagenet)
        batch_manager = CentralBatchManager(dataset=dataset, args=args)
        dataset_info = batch_manager.dataset_info()

        num_batches = dataset_info['num_batches']


        simulator = NextEventSimulator(
        batch_manager=batch_manager,
        job_speeds= job_speeds,
        batch_size = 128,
        batches_per_epoch = num_batches,
        epochs_per_job = epochs_per_job)

        simulator.run_simulation()
        simulator.report_results(job_speeds, epochs_per_job, batch_size, partion_size)
#plot the number of batches in the system over time
# import matplotlib.pyplot as plt
# plt.plot(list(simulator.batches_in_system_over_time.keys()), list(simulator.batches_in_system_over_time.values()))
# plt.xlabel('Time')
# plt.ylabel('Number of Batches in System')
# plt.title('Number of Batches in System Over Time')
# plt.show()

#0.00003125