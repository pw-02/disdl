import threading
from collections import deque, OrderedDict
from typing import  Dict, Tuple
from common.batch import Batch
import time
from common.logger_config import logger
from common.utils import AverageMeter
import concurrent.futures
import boto3
import json
from botocore.config import Config
from datetime import datetime, timedelta, timezone
from common.job import DLTJob
import math
from typing import OrderedDict as TypingOrderedDict
from common.dataset import Dataset
from concurrent.futures import ThreadPoolExecutor

class PrefetchService:
    def __init__(self, 
                 lambda_name: str, 
                 cache_address: str, 
                 jobs:Dict[str, DLTJob], 
                 dataset: Dataset,
                 cost_threshold_per_hour: float = None,
                 simulate_time: float = None):
        
        self.lambda_name = lambda_name
        self.jobs = jobs
        self.lambda_client = boto3.client("lambda")
        self.lambda_invocations_count = 0
        self.lock = threading.Lock()
        self.prefetch_stop_event = threading.Event()
        self.prefetch_cycle_times = AverageMeter('Prefetch Cycle Time')
        self.executor = ThreadPoolExecutor(max_workers=5)  # Reuse thread pool
        self.lambda_execution_times = AverageMeter('Lambda Execution Time')
        self.simulate_time:float = simulate_time
        self.prefetch_delay:float = 0
        self.cost_threshold_per_hour = cost_threshold_per_hour
        self.dataset:Dataset = dataset
        self.cache_address = cache_address
        self.prefetch_lambda_configured_memory = self.get_memory_allocation_of_lambda()

    def get_memory_allocation_of_lambda(self):
        response = self.lambda_client.get_function_configuration(FunctionName=self.lambda_name)
        return response.get("MemorySize", 1024)  # Default to 1024MB if not found
    
    def _prefetching_process(self):
        while not self.prefetch_stop_event.is_set():
            try:
                prefetch_cycle_start_time = time.perf_counter()
                prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]] = OrderedDict()
                prefetch_cycle_duration = self.lambda_execution_times.avg + self.prefetch_delay if self.lambda_execution_times.count > 0 else self.simulate_time if self.simulate_time else 2.5

                for job in self.jobs.values():

                    prefetch_conncurrency = job.compute_required_prefetch_concurrency(prefetch_cycle_duration, buffer=5)
                    
                    if prefetch_conncurrency <= 1:
                        logger.info(f"Job '{job.job_id}' requires no prefetching. Required_prefetch_bacthes_per_second: {prefetch_conncurrency}")
                        continue
                    
                    if len(job.future_batches) < prefetch_conncurrency:
                        logger.info(f"Job '{job.job_id}' has {len(job.future_batches)} batches, less than the required prefetch concurrency of {prefetch_conncurrency}.")

                    prefetch_counter = 0
                    time_counter = 0
                    job_batches_snapshot = list(job.future_batches.values())
                    for batch in job_batches_snapshot:
                        if time_counter <= prefetch_cycle_duration:
                                # If accessed within the cycle, add its time to the counter
                                if batch.is_cached or batch.caching_in_progress:
                                    time_counter += job.get_gpu_batch_rate()
                                else:
                                    time_counter += job.get_gpu_batch_rate() + job.get_data_loading_delay()
                                logger.info(f"batch '{batch.batch_id}' wont be prefetched in time. Skipping.")
                                continue
                        
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

    
    def start_prefetcher(self):
        self.start_time = time.perf_counter()
        self.prefetch_stop_event.clear()
        prefetch_thread = threading.Thread(target=self._prefetching_process)
        prefetch_thread.daemon = True
        prefetch_thread.start()

    def prefetch_batches_from_list(self, prefetch_list: TypingOrderedDict[str, Tuple[Batch, str]], 
                                   prefetch_cycle_start_time: float, 
                                   is_warm_up: bool = False):
        
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
                    
                    self.lambda_execution_times.update(response['execution_time'])
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
        self.lambda_invocations_count += len(prefetch_list)
        
        if not is_warm_up:
            self.prefetch_cycle_times.update(time.perf_counter() - prefetch_cycle_start_time + delay_time)

    def _prefetch_batch(self, payload):
            if self.simulate_time:
                return {'success': True, 'message': None, 'execution_time': self.simulate_time}
            
            request_started = time.perf_counter()
            response = self.prefetch_lambda_client.invoke(FunctionName=self.lambda_name,
                                                          InvocationType='RequestResponse',
                                                          Payload=payload)
            
            response_data = json.loads(response['Payload'].read().decode('utf-8'))
            response_data['execution_time'] = time.perf_counter() - request_started
            return response_data
    
    
    def _compute_prefetch_lambda_request_rate(self):
        with self.lock:
            elapsed_time = time.perf_counter() - self.start_time  # Calculate elapsed time
            if elapsed_time > 0:
                request_rate = self.lambda_invocations_count / elapsed_time  # Compute request rate
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
            
            cost_per_request =  current_prefetch_cost / self.lambda_invocations_count
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
        

    def stop_prefetcher(self):
            if not self.prefetch_stop_event.is_set():
                self.prefetch_stop_event.set()
                # logger.info(f"Prefetcher stopped. Total requests: {self.prefetch_lambda_invocations_count}, Total execution time: {self.prefetch_lambda_execution_times.sum:.2f}s, Total cost: ${self._compute_prefeteching_cost():.4f}")
                logger.info(f"Prefetching stopped")


    def _compute_prefeteching_cost(self):    
        # AWS Lambda pricing details (replace these with the latest rates from AWS)
        REQUEST_COST_PER_MILLION = 0.20  # USD per million requests
        GB_SECOND_COST = 0.0000166667  # USD per GB-second

        # Compute request cost
        request_cost = (self.prefetch_lambda_invocations_count / 1_000_000) * REQUEST_COST_PER_MILLION

        # Convert memory from MB to GB
        memory_gb = self.prefetch_lambda_configured_memory / 1024

        # Calculate total GB-seconds
        compute_time_gb_seconds = memory_gb * self.lambda_execution_times.sum

        # Calculate compute cost
        compute_cost = compute_time_gb_seconds * GB_SECOND_COST

        # Total cost
        total_cost = request_cost + compute_cost
        return total_cost
    
if __name__ == "__main__":
    
    from dataset import Dataset
    dataset = Dataset(
        data_dir="s3://sdl-cifar10/test/",
        transforms=None,
        max_dataset_size=None,
        use_local_folder=False)
    print(f"Total samples: {len(dataset)}")