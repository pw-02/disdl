import threading
from collections import deque, OrderedDict
from typing import  Dict, Tuple
from batch import Batch
import time
from logger_config import logger
from utils import AverageMeter
import concurrent.futures
import boto3
import json
from botocore.config import Config
from datetime import datetime, timedelta, timezone
from job import DLTJob
import math
from typing import OrderedDict as TypingOrderedDict
from dataset import Dataset

class PrefetchService:
    def __init__(self, 
                 prefetch_lambda_name: str, 
                 cache_address: str, 
                 jobs:Dict[str, DLTJob], 
                 dataset: Dataset,
                 cost_threshold_per_hour: float = 10,
                 simulate_time: float = None):
        
        self.prefetch_lambda_execution_times = AverageMeter('Lambda Execution Time')
        self.prefetch_cycle_times = AverageMeter('Prefetch Cycle Time')
        self.prefetch_stop_event:threading.Event = threading.Event()
        self.prefetch_stop_event.set()
        self.cache_address:str = cache_address
        self.simulate_time:float = simulate_time
        self.lock = threading.Lock()
        self.jobs:Dict[str, DLTJob] = jobs
        self.prefetch_delay:float = 0
        self.cost_threshold_per_hour = cost_threshold_per_hour
        self.start_time = None
        self.start_time_utc = datetime.now(timezone.utc)
        self.dataset = dataset

        self.prefetch_lambda_name = prefetch_lambda_name
        self.prefetch_lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))
        self.prefetch_lambda_configured_memory = self.get_memory_allocation_of_lambda(self.prefetch_lambda_name)
        self.prefetch_lambda_invocations_count = 0

        # self.prefetch_lambda_execution_times.update(get_average_lambda_duration(self.prefetch_lambda_name))
        pass

    def get_memory_allocation_of_lambda(self):
        client = boto3.client('lambda')
        response = client.get_function_configuration(FunctionName=self.prefetch_lambda_name)
        return response['MemorySize']
    
    
    def start_prefetcher(self):
        self.start_time = time.perf_counter()
        self.prefetch_stop_event.clear()
        prefetch_thread = threading.Thread(target=self._prefetching_process)
        prefetch_thread.daemon = True
        prefetch_thread.start()

    def _compute_opttiomal_prefetch_lookahead(self, data_delay, job_gpu_time):

        #training job can process at batch in 'job_gpu_time' seconds, so in 'data_delay' seconds it can process..
        potential_batches_during_delay = data_delay / job_gpu_time #this is the number of batches that can could be processed duing the delay
        #o ensure there is no delay, the job should be able to process at least 'potential_batches' within delay time
        #if it takes the prefecher 'sef.prefecth_cycle_time' to prfetch a batch, then requrired concurrency is...
        batches_loaded_per_conccurency_unit =  data_delay / self.prefetch_cycle_times.avg
        required_concurrency = potential_batches_during_delay / batches_loaded_per_conccurency_unit 
        return required_concurrency
        
    def prefetch_batches_from_list(self, prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]], prefetch_cycle_start_time: float, is_warm_up: bool = False):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create client if not already initialized
            if self.prefetch_lambda_client is None:
                self.prefetch_lambda_client = boto3.client('lambda', config= Config(max_pool_connections=50))

            # Calculate the delay before invoking Lambdas
            delay_time = self._compute_delay_to_satisfy_cost_threshold() * len(prefetch_list)

            if delay_time > 0:
                logger.info(f"Delaying prefetching by {delay_time:.5f} seconds to satisfy cost threshold.")
                time.sleep(delay_time)
            
            if self.simulate_time:
                time.sleep(self.simulate_time)

            # Map each future to its corresponding (batch, payload) tuple
            future_to_batch_payload = {executor.submit(self._prefetch_batch, payload): (batch, payload) for batch, payload in prefetch_list.values()}

            for future in concurrent.futures.as_completed(future_to_batch_payload):
                batch, payload = future_to_batch_payload[future]
                try:
                    response = future.result()

                    batch.set_caching_in_progress(in_progress=False)
                    
                    self.prefetch_lambda_execution_times.update(response['execution_time'])
                    # self.prefetch_lambda_invocations_count += 1
                    
                    if 'success' in response.keys() and response['success']:
                        # print(f"Batch '{batch.batch_id}' has been prefetched.")
                        batch.set_cache_status(is_cached=True)
                        batch.set_last_accessed_time()
                        batch.prefetched_time_utc = datetime.now(timezone.utc)
                    else:
                        batch.set_cache_status(is_cached=False)
                        if 'message' in response.keys():
                            logger.error(f"Error prefetching batch '{batch.batch_id}': {response['message']}")
                        else:
                            logger.error(f"Error prefetching batch '{batch.batch_id}'.")
                    # print(f'Invocation response: {response}')
                except Exception as e:
                    logger.error(f"Error in prefetching batch: {e}", exc_info=True)
        self.prefetch_lambda_invocations_count += len(prefetch_list)
        
        if not is_warm_up:
            self.prefetch_cycle_times.update(time.perf_counter() - prefetch_cycle_start_time + delay_time)

    def _prefetching_process(self):
        while not self.prefetch_stop_event.is_set():
            try:
                prefetch_cycle_start_time = time.perf_counter()
                prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]] = OrderedDict()
                #prefetch_cycle_duration = self.prefetch_cycle_times.avg if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
                for job in self.jobs.values():
                    if job.total_steps <= 1: #ignore first two steps for GPU warm up
                        continue

                    max_bacthes_per_second = math.ceil(1 / job.training_step_gpu_times.avg)
                    no_caching_batches_per_second =  math.floor(1 / job.dataload_time_on_miss.avg) if job.dataload_time_on_miss.count > 0 else 0
                    required_prefetch_bacthes_per_second = max_bacthes_per_second - no_caching_batches_per_second

                    if required_prefetch_bacthes_per_second < 1:
                        logger.info(f"Job '{job.job_id}' requires no prefetching. required_prefetch_bacthes_per_second: {required_prefetch_bacthes_per_second}")
                        continue
                        # required_prefetch_bacthes_per_second=5
                    prefetch_cycle_duration = self.prefetch_lambda_execution_times.avg + self.prefetch_delay if self.prefetch_lambda_execution_times.count > 0 else self.simulate_time if self.simulate_time else 2.5
                    
                    #prefetch_cycle_duration = self.prefetch_cycle_times.avg + self.prefetch_delay if self.prefetch_cycle_times.count > 0 else self.simulate_time if self.simulate_time else 3
                    prefetch_conncurrency =  math.ceil(required_prefetch_bacthes_per_second * prefetch_cycle_duration) + 5 #add a buffer of 5

                    logger.info(f'prefetch_conncurrency: {prefetch_conncurrency}, prefetch_cycle_duration: {prefetch_cycle_duration}, required_prefetch_bacthes_per_second: {required_prefetch_bacthes_per_second}')
                    #add in a check to see if the job is suffering from a data loading delay and benefit from prefetching
                    prefetch_counter, time_counter = 0, 0
                    # Fetch average times for cache hit and miss scenarios for the current job
                    ave = job.training_step_times_on_hit.avg if job.training_step_times_on_hit.count > 0 else job.training_step_gpu_times.avg
                    avg_time_on_miss = job.training_step_times_on_miss.avg if job.training_step_times_on_miss.count > 0 else job.training_step_gpu_times.avg + 1.5

                    if len(job.future_batches) < prefetch_conncurrency:
                        logger.info(f"Job '{job.job_id}' has {len(job.future_batches)} batches, less than the required prefetch concurrency of {prefetch_conncurrency}.")

                    # Iterate over future batches to determine access during the prefetch cycle duration
                    job_batches_snapshot = list(job.future_batches.values())
                    for batch in job_batches_snapshot:
                        # if time_counter <= prefetch_cycle_duration:
                        #         # If accessed within the cycle, add its time to the counter
                        #         if batch.is_cached or batch.caching_in_progress:
                        #             time_counter += avg_time_on_hit
                        #         else:
                        #             time_counter += avg_time_on_miss
                        #         logger.info(f"batch '{batch.batch_id}' wont be prefetched in time. Skipping.")
                        #         continue
                        
                        if prefetch_counter >= prefetch_conncurrency:
                            break

                        else: 
                            prefetch_counter += 1
                            if not batch.is_cached and not batch.caching_in_progress:
                                # prefetch_counter += 1
                                logger.debug(f"prefetching batch '{batch.batch_id}'")

                                batch.set_caching_in_progress(True)
                                payload = {
                                    'bucket_name': self.dataset.bucket_name,
                                    'batch_id': batch.batch_id,
                                    'batch_samples': self.dataset.get_samples(batch.indicies),
                                    'cache_address': self.cache_address,
                                    'task': 'prefetch',
                                }
                                prefetch_list[batch.batch_id] = (batch,json.dumps(payload))
                                # prefetch_list.add((batch, json.dumps(payload)))
                            else:
                                logger.debug(f"batch '{batch.batch_id}' is already being prefetched")
                  
                # Submit the prefetch list for processing
                if prefetch_list:
                    logger.info(f"Prefetching {len(prefetch_list)} batches for {prefetch_conncurrency} concurrency.")
                    self.prefetch_batches_from_list(prefetch_list, prefetch_cycle_start_time)
                    logger.info(f"Prefetch took: {self.prefetch_cycle_times.val:.4f}s for {len(prefetch_list)} batches. (Avg Prefetch Time: {self.prefetch_cycle_times.avg:.4f}s, Avg Lambda Time: {self.prefetch_lambda_execution_times.avg:.4f}s, Running Cost: ${self._compute_prefeteching_cost():.4f})")

                if len(self.jobs) == 0 or len(prefetch_list) == 0:
                    time.sleep(0.1)  # Sleep for a short while before checking again

            except Exception as e:
                logger.error(f"Unexpected error in prefetching process: {e}", exc_info=True)

 

    def _prefetch_batch(self, payload):
            if self.simulate_time:
                return {'success': True, 'message': None, 'execution_time': self.simulate_time}
            
            request_started = time.perf_counter()
            response = self.prefetch_lambda_client.invoke( FunctionName=self.prefetch_lambda_name,
                                                InvocationType='RequestResponse',
                                                Payload=payload)
            
            response_data = json.loads(response['Payload'].read().decode('utf-8'))
            response_data['execution_time'] = time.perf_counter() - request_started
            return response_data
    
    def _get_average_request_duration(self):
        pass
        # if start_time is not None:
        #     return get_average_lambda_duration(self.name, start_time=start_time, end_time = datetime.now(timezone.utc)) #average duration since prefetching started
        # else:
        #     return get_average_lambda_duration(self.name, 
        #                                        start_time=datetime.now(timezone.utc)  - timedelta(hours=4), #average duration for the last 4 hours
        #                                        end_time = datetime.now(timezone.utc))
    
    def _compute_prefetch_lambda_request_rate(self):
        with self.lock:
            elapsed_time = time.perf_counter() - self.start_time  # Calculate elapsed time
            if elapsed_time > 0:
                request_rate = self.prefetch_lambda_invocations_count / elapsed_time  # Compute request rate
                return request_rate
            return 0.0
        
    def _compute_delay_to_satisfy_cost_threshold(self):
        """Calculate delay based on current cost and the predefined cost threshold."""
        if self.cost_threshold_per_hour is None:
            return 0
        else:
            #  avg_request_duration = self.prefetch_lambda.get_average_request_duration(self.start_time_utc)
            current_prefetch_cost = self._compute_prefeteching_cost()
            if current_prefetch_cost == 0:
                return 0
            # logger.info(f"Current prefetch cost: {current_prefetch_cost:.16f}, Requests: {self.prefetch_lambda_invocations_count}, Execution time: {self.prefetch_lambda_execution_times.sum:.2f} seconds")
            
            cost_per_request =  current_prefetch_cost / self.prefetch_lambda_invocations_count
            request_rate = self._compute_prefetch_lambda_request_rate() 
            requests_per_hour = request_rate * 3600  # Convert request rate to requests per hour
            # Calculate the maximum allowable requests per hour within the cost threshold
            max_requests_per_hour = self.cost_threshold_per_hour / cost_per_request
            # If the current request rate is within the threshold, no delay is needed
            if requests_per_hour <= max_requests_per_hour:
                return 0  # No delay needed
    
            # Calculate the required delay in hours between each request to stay within the cost threshold
            delay_per_request_hours = (1 / max_requests_per_hour) - (1 / requests_per_hour)
            # Convert delay from hours to seconds for practical use
            delay_per_request_seconds = delay_per_request_hours * 3600
            self.prefetch_delay = max(delay_per_request_seconds, 0)
            return max(delay_per_request_seconds, 0)
        
    def _compute_prefeteching_cost(self):
        current_prefetch_cost = self.compute_lambda_cost()
        return current_prefetch_cost
    
    def stop_prefetcher(self):
            if not self.prefetch_stop_event.is_set():
                self.prefetch_stop_event.set()
                logger.info(f"Prefetcher stopped. Total requests: {self.prefetch_lambda_invocations_count}, Total execution time: {self.prefetch_lambda_execution_times.sum:.2f}s, Total cost: ${self._compute_prefeteching_cost():.4f}")

    def compute_lambda_cost(self):    
        # AWS Lambda pricing details (replace these with the latest rates from AWS)
        REQUEST_COST_PER_MILLION = 0.20  # USD per million requests
        GB_SECOND_COST = 0.0000166667  # USD per GB-second

        # Compute request cost
        request_cost = (self.prefetch_lambda_invocations_count / 1_000_000) * REQUEST_COST_PER_MILLION

        # Convert memory from MB to GB
        memory_gb = self.prefetch_lambda_configured_memory / 1024

        # Calculate total GB-seconds
        compute_time_gb_seconds = memory_gb * self.prefetch_lambda_execution_times.sum

        # Calculate compute cost
        compute_cost = compute_time_gb_seconds * GB_SECOND_COST

        # Total cost
        total_cost = request_cost + compute_cost
        return total_cost