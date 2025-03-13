import heapq
import logging
import numpy as np
from typing import List, Dict
import csv
import os
# from config import sim_config
def save_dict_list_to_csv(dict_list, output_file):
    if not dict_list:
        print("No data to save.")
        return
    headers = dict_list[0].keys()
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=',')
        if not file_exists:
            writer.writeheader()
        for data in dict_list:
            writer.writerow(data)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),logging.FileHandler("simulation.log", mode="w")])

class TensorSoketCache:
    def __init__(self, total_jobs):
        self.cache = {}
        self.cache_size_over_time = []
        self.total_jobs = total_jobs
        self.size_of_cache_object_gb =  0.039
        #Track cache size at each step 

    def insert_bacth(self, new_batch,):
        if new_batch not in self.cache:
            self.cache[new_batch] = 0
            self.cache_size_over_time.append(len(self.cache))

    def get_batch (self, requested_batch):
        cache_hit = False
        if requested_batch in self.cache:
            cache_hit = True
            self.cache[requested_batch] += 1
            access_count = self.cache[requested_batch]
            if access_count + 1 >= self.total_jobs:
                del self.cache[requested_batch]
        return cache_hit


    def get_cache_len(self):
        return len(self.cache)
    
    def get_cache_size_gb(self):
        return len(self.cache) *  self.size_of_cache_object_gb
    
    def get_max_num_of_cache_items(self):
        return max(self.cache_size_over_time)
    
    def get_avg_num_of_cache_items(self):
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time)
    
    def get_max_cache_size_gb(self):
        return  max(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
    def get_avg_cache_size_gb(self):
        
        return sum(self.cache_size_over_time) / len(self.cache_size_over_time) *  self.size_of_cache_object_gb
    
class Producer:
    def __init__(self, producer_id, time_to_load_batch, batches_per_epoch, total_epochs, consumer_buffer_size, cache:TensorSoketCache):
        self.producer_id = producer_id
        self.time_to_load_batch = time_to_load_batch
        self.total_epochs_to_process = total_epochs
        self.total_batches_per_epoch = batches_per_epoch
        self.batches_remaining = list(range(1, (total_epochs * batches_per_epoch) + 1))
        self.batches_processed_count = 0
        self.epochs_processed_count = 0
        self.next_available_time = 0
        self.cache = cache
        self.cache_hits = []
        self.cache_misses = []
        self.batch_size = 128
        self.consumer_buffer_size = consumer_buffer_size
        self.next_available_time_on_miss = 0.01

    def __lt__(self, other):
        return self.next_available_time < other.next_available_time
    
    def process_next_batch(self):
        if not self.batches_remaining:
            return False  # No more batches to process, prevent popping an empty list
        
        if self.cache.get_cache_len() > self.consumer_buffer_size: # if cache is full, wait for a consumer to consume a batch
            self.next_available_time += 0.001
        else:
            next_batch = self.batches_remaining.pop(0)
            self.cache.insert_bacth(next_batch)
            self.next_available_time += self.time_to_load_batch


class Consumer:
    def __init__(self, consumer_id, time_per_batch, batches_per_epoch, total_epochs, cache:TensorSoketCache):
        self.consumer_id = consumer_id
        self.time_per_batch = time_per_batch  
        self.total_epochs_to_process = total_epochs
        self.total_batches_per_epoch = batches_per_epoch
        self.batches_remaining = list(range(1, (total_epochs * batches_per_epoch) + 1))
        self.batches_processed_count = 0
        self.epochs_processed_count = 0
        self.next_available_time = 0 
        self.cahe_hits = []
        self.cache_misses = []
        self.next_available_time_on_miss = 0.01
        self.cache = cache
        self.batch_size = 128
    
    def __lt__(self, other):
        return self.next_available_time < other.next_available_time  
    
    def process_next_batch(self):
        if not self.batches_remaining:
            return False  # No more batches to process, prevent popping an empty list
        #peek at the next batch
        next_batch = self.batches_remaining[0]
        is_cache_hit = self.cache.get_batch(next_batch)
        if not is_cache_hit:
            #if not already recorded as a miss, record it now
            if next_batch not in self.cache_misses:
                self.cache_misses.append(next_batch)
            self.next_available_time += self.next_available_time_on_miss
        else:
            #if not already recorded as a hit, record it now
            if next_batch not in self.cahe_hits:
                self.cahe_hits.append(next_batch)
            self.next_available_time += self.time_per_batch
            next_batch = self.batches_remaining.pop(0)
        
        if self.batches_processed_count % self.total_batches_per_epoch == 0:
            self.epochs_processed_count += 1
            # self.cache_size_over_time.append(self.cache.get_cache_len())
    
    def gen_perf_metrics(self):
        line = {}
        line['job_id'] = self.consumer_id
        line['epochs_processed'] = self.epochs_processed_count
        line['batches_processed'] = self.batches_processed_count
        line['samples_processed'] = self.batches_processed_count * self.batch_size
        line['time_per_batch'] = self.time_per_batch
        line['cache_hits'] = len(self.cahe_hits)
        line['cache_misses'] = len(self.cache_misses)
        line['cache_hit_rate'] = len(self.cahe_hits) / (len(self.cahe_hits) + len(self.cache_misses))
        line['potential_time(sec)'] = self.batches_processed_count * self.time_per_batch
        line['potential_time(hour)'] = self.batches_processed_count * self.time_per_batch / 3600 # 1 hour = 3600 seconds
        line['actual_time(sec)'] = self.next_available_time
        line['actual_time(hour)'] = self.next_available_time / 3600 # 1 hour = 3600 seconds
        line['potential_throughput(batches/sec)'] = self.batches_processed_count / (self.batches_processed_count * self.time_per_batch)
        line['actual_throughput(batches/sec)'] = self.batches_processed_count / self.next_available_time
        line['potential_throughput(samples/sec)'] = self.batches_processed_count / (self.batches_processed_count * self.time_per_batch) *  self.batch_size
        line['actual_throughput(samples/sec)'] = self.batches_processed_count / self.next_available_time * self.batch_size
        line['dataloading_delay (sec)'] = line['actual_time(sec)'] - line['potential_time(sec)']
        line['dataloading_delay (hour)'] = line['actual_time(hour)'] - line['potential_time(hour)']
        return line

def run(config):
    batches_per_epoch = config['batches_per_epoch']
    total_epochs = config['total_epochs']
    consumer_speeds = config['job_speeds']
    producer_speed = config['producer_time_to_load_batch']
    consumer_buffer_size = config['consumer_buffer_size']
    cache = TensorSoketCache(len(consumer_speeds))
    consumers: List[Consumer] = []
    process_queue = []

    producer = Producer(0, producer_speed, batches_per_epoch, total_epochs, consumer_buffer_size, cache)
    heapq.heappush(process_queue, producer)

    for consumer_id, time_per_batch in enumerate(consumer_speeds):
        consumer = Consumer(consumer_id, time_per_batch, batches_per_epoch, total_epochs, cache)
        consumers.append(consumer)
        heapq.heappush(process_queue, consumer)

    training_finished = False

    while not training_finished:
        training_finished = True  # Assume training is done unless proven otherwise
        if process_queue:
            training_finished = False  # There are jobs still running
            process = heapq.heappop(process_queue)  # Get the process with the earliest available time - either job or producer
            if isinstance(process, Producer):
                if len(process.batches_remaining) == 0:
                    logging.info(f"Producer {process.producer_id} has no more batches to process.")
                    continue
                process.process_next_batch()
            
            if isinstance(process, Consumer):
                if len(process.batches_remaining) == 0:
                    logging.info(f"Job {process.consumer_id} has no more batches to process.")
                    continue
                process.process_next_batch()

        heapq.heappush(process_queue, process)

    # for job in consumers:
    #     total_delay = sum(delay for delay in job.epoch_delays.values())
    #     logging.info(f"Total delay for Job {job.job_id} due to coordl policy: {total_delay:.2f} seconds")
    
    # for job in jobs:
    #     throughput = job.batches_processed / job.next_available_time
    #     logging.info(f"Throughput for Job {job.job_id}: {throughput:.2f} batches per second")

    logging.info(f"max_cache_size_gb: {cache.get_max_num_of_cache_items()}")
    logging.info(f"max_cache_size_gb: {cache.get_max_cache_size_gb()}")

    final_metrics = [job.compute_final_metrics() for job in consumers]
    save_dict_list_to_csv(final_metrics, "sim_final_metrics.csv")
    summary = {}
    summary['dataset_size(num_batches)'] = config['batches_per_epoch'] * config['total_epochs']
    summary['total_jobs'] = len(consumers)
    summary['epochs_per_job'] = total_epochs
    summary['total_epochs'] = sum(job['epochs_processed'] for job in final_metrics)
    summary['total_batches_processed'] = sum(job['batches_processed'] for job in final_metrics)
    summary['total_samples_processed'] = sum(job['samples_processed'] for job in final_metrics)
    summary['toal_time(sec)'] = sum(job['actual_time(sec)'] for job in final_metrics) / len(final_metrics)
    summary['total_time(hours)'] =sum(job['actual_time(hour)'] for job in final_metrics) / len(final_metrics)
    summary['total_throughput(samples/sec)'] = sum(job['actual_throughput(samples/sec)'] for job in final_metrics)
    summary['total_throughput(bacthes/sec)'] = sum(job['actual_throughput(batches/sec)'] for job in final_metrics)
    summary['potential_throughput(samples/sec)'] = sum(job['potential_throughput(samples/sec)'] for job in final_metrics)
    summary['potential_throughput(bacthes/sec)'] = sum(job['potential_throughput(batches/sec)'] for job in final_metrics) 

    # summary['total_throughput(samples/sec)'] = summary['total_samples_processed']/ summary['toal_time(sec)']
    # summary['total_throughput(bacthes/sec)'] = summary['total_batches_processed']/ summary['toal_time(sec)']
    summary['compute_cost'] = summary['total_time(hours)'] * 12.24  # $0.1 per hour
    summary['max_number_of_cached_bacthes'] = cache.get_max_num_of_cache_items()
    summary['max_cache_size_gb'] = cache.get_max_cache_size_gb()
    summary['avg_number_of_cached_bacthes'] = cache.get_avg_num_of_cache_items()
    summary['avg_cache_size_gb'] = cache.get_avg_cache_size_gb()
    summary['cache_hits'] = sum(job['cache_hits'] for job in final_metrics)
    summary['cache_misses'] = sum(job['cache_misses'] for job in final_metrics)
    summary['cache_hit_rate'] = summary['cache_hits'] / (summary['cache_hits'] + summary['cache_misses'])
    summary['total_cost'] = summary['compute_cost']
    save_dict_list_to_csv([summary], "tensorysocket_sim_metrics.csv")

if __name__ == "__main__":
    np.random.seed(42)

    config :Dict = {
        'batches_per_epoch': 100,
        'total_epochs': 3,
        'job_speeds': [1, 2, 4],
        'producer_speed': 3,
        'consumer_buffer_size': 10,
        'producer_time_to_load_batch:': 2
        }
    
    if  os.path.isfile("sim_final_metrics.csv"):
        os.remove("sim_final_metrics.csv")
    if  os.path.isfile("sim_final_summary_metrics.csv"):
        os.remove("sim_final_summary_metrics.csv")
    run(config)

print("Simulation completed successfully!")